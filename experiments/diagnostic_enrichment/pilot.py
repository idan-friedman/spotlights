"""End-to-end profiler-guided IOCR diagnostic-enrichment pilot.

Run from the Spotlights repository root with the IOCR-capable Python environment.
The subject checkout is read-only; only the chosen recognition boundary is captured.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.diagnostic_enrichment.contract import (
    SCHEMA_VERSION,
    atomic_bytes,
    atomic_json,
    file_sha256,
    object_sha256,
    read_json,
)
from experiments.diagnostic_enrichment.recognition_bucket import (
    ADAPTER_NAME,
    SOURCE,
    run_subject,
)
from experiments.diagnostic_enrichment.recognition_bucket import (
    replay as replay_recognition_bucket,
)

PINNED_IOCR_COMMIT = "641b9806fcc38a33180c60396ac0b8df9a2b19e6"
ADAPTERS = {
    ADAPTER_NAME: {"capture": run_subject, "replay": replay_recognition_bucket},
}
DIRECT_SOURCE = "hrl_ocr/models/fonts_style_recognition/font_size_estimator.py"


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.iocr_repo.resolve()
    images_dir = args.images_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.is_relative_to(repo) or run_dir.is_relative_to(images_dir):
        raise ValueError("run artifacts must be outside the IOCR checkout and image workload")
    if _git(repo, "rev-parse", "HEAD") != PINNED_IOCR_COMMIT:
        raise ValueError("IOCR checkout is not at pinned commit")
    if _git(repo, "status", "--porcelain"):
        raise ValueError("IOCR checkout is dirty")
    if not os.environ.get("IOCR_WEIGHTS_LOCATION"):
        raise ValueError("IOCR_WEIGHTS_LOCATION is required")
    images = sorted(images_dir.glob("*.jpg"))
    if len(images) != 44:
        raise ValueError("expected the pinned 44-page JPEG workload")
    manifest = read_json(args.profile_manifest)
    if manifest["iocr_git"]["sha"] != PINNED_IOCR_COMMIT:
        raise ValueError("profile subject commit differs")
    if Path(manifest["images_dir"]).resolve() != images_dir or manifest["languages"] != ["eng"]:
        raise ValueError("profile workload or language differs")
    weights_root = Path(os.environ["IOCR_WEIGHTS_LOCATION"]).resolve()
    if weights_root != Path(manifest["iocr_weights_location"]).resolve():
        raise ValueError("IOCR weights location differs from pinned profile configuration")
    workload = [
        {"name": image.name, "bytes": image.stat().st_size, "sha256": file_sha256(image)}
        for image in images
    ]
    return {
        "repo": repo,
        "images_dir": images_dir,
        "weights_root": weights_root,
        "workload": workload,
        "workload_sha256": object_sha256(workload),
        "source_sha256": file_sha256(repo / SOURCE),
    }


def _profile_hit(
    lines: list[str], name: str, location: str, *, allow_multiple: bool = False
) -> dict[str, Any]:
    matches = [
        (index + 1, line)
        for index, line in enumerate(lines)
        if name in line and location in line and re.search(r"\d+\.\d+ " + re.escape(name), line)
    ]
    if not matches or (len(matches) != 1 and not allow_multiple):
        raise ValueError(f"unexpected profile hits for {name} at {location}: {len(matches)}")
    number, line = matches[0]
    cost = re.search(r"(\d+\.\d+) " + re.escape(name), line)
    assert cost is not None
    return {"line": number, "duration_s": float(cost.group(1)), "call": name, "location": location}


def _source_line(repo: Path, relative: str, needle: str) -> int:
    lines = (repo / relative).read_text().splitlines()
    found = [index + 1 for index, line in enumerate(lines) if needle in line]
    if len(found) != 1:
        raise ValueError(f"expected one source occurrence of {needle!r} in {relative}")
    return found[0]


def _intake(args: argparse.Namespace, context: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    profile = args.profile.resolve()
    lines = profile.read_text(errors="replace").splitlines()
    log_lines = args.profile_log.read_text(errors="replace").splitlines()
    processed_images = [
        Path(line.split("Processing file ", 1)[1].strip()).name
        for line in log_lines
        if "Processing file " in line
    ]
    workload_names = {item["name"] for item in context["workload"]}
    if len(processed_images) != 44 or set(processed_images) != workload_names:
        raise ValueError("profile log does not cover the frozen 44-page workload")
    inner_rows = read_json(args.inner_profile)["per_image_timing_info"]
    if len(inner_rows) != 44:
        raise ValueError("inner profiler does not cover 44 pages")
    inner_costs = {
        "pipeline_runtime_s": sum(row["pipeline_runtime"] for row in inner_rows),
        "recognition_s": sum(row["activities"]["recognition"]["runtime"] for row in inner_rows),
        "fonts_style_size_s": sum(
            row["activities"]["fonts_style_size"]["runtime"] for row in inner_rows
        ),
    }
    duration = next(
        (
            float(match.group(1))
            for line in lines[:8]
            if (match := re.search(r"Duration:\s*([\d.]+)", line))
        ),
        None,
    )
    if duration is None:
        raise ValueError("profile has no wall duration")
    font = _profile_hit(
        lines, "FontSizeEstimator.normalize_bboxes_info_list", "font_size_estimator.py:237"
    )
    if font["line"] >= len(lines):
        raise ValueError("font-size profile branch is incomplete")
    copy_line = font["line"] + 1
    copy_match = re.search(r"(\d+\.\d+) deepcopy  copy.py:118", lines[copy_line - 1])
    if copy_match is None:
        raise ValueError("font-size call has no direct deepcopy child in profile")
    copy_cost = float(copy_match.group(1))
    recognition = _profile_hit(
        lines, "RecognizerByScript.recognize_according_to_script", "recognizer_by_script.py:26"
    )
    onnx = _profile_hit(
        lines,
        "InferenceSession.run",
        "onnxruntime_inference_collection.py:304",
        allow_multiple=True,
    )
    direct_line = _source_line(
        context["repo"], DIRECT_SOURCE, "boxes_to_normalize = copy.deepcopy(boxes_info_list)"
    )
    bucket_line = _source_line(
        context["repo"], SOURCE, "for width_idx, width_batch in enumerate(imgW[:-1]):"
    )
    fallthrough_line = _source_line(context["repo"], SOURCE, "if not crop_was_placed:")
    resize_line = _source_line(context["repo"], SOURCE, "resized_w = math.ceil(imgH * ratio)")
    pad_source = "hrl_ocr/models/recognition/ssocr/dataset.py"
    pad_line = _source_line(context["repo"], pad_source, "Pad_img[:, :, w:] = img[:, :, w - 1]")
    intake = {
        "schema_version": SCHEMA_VERSION,
        "subject_commit": PINNED_IOCR_COMMIT,
        "subject_root": str(context["repo"]),
        "profile": {"path": str(profile), "sha256": file_sha256(profile), "wall_s": duration},
        "profile_log": {
            "path": str(args.profile_log.resolve()),
            "sha256": file_sha256(args.profile_log),
            "processed_pages": len(processed_images),
            "covers_frozen_workload": True,
        },
        "inner_profile": {
            "path": str(args.inner_profile.resolve()),
            "sha256": file_sha256(args.inner_profile),
            "per_image_records": len(inner_rows),
            "costs": inner_costs,
        },
        "profile_manifest": {
            "path": str(args.profile_manifest.resolve()),
            "sha256": file_sha256(args.profile_manifest),
        },
        "workload": {
            "images_dir": str(context["images_dir"]),
            "pages": len(context["workload"]),
            "sha256": context["workload_sha256"],
            "files": context["workload"],
            "languages": ["eng"],
            "device": "CPU",
        },
        "runtime_configuration": {
            "weights_root": str(context["weights_root"]),
            "device": "CPU",
            "languages": ["eng"],
        },
        "source": {
            SOURCE: file_sha256(context["repo"] / SOURCE),
            DIRECT_SOURCE: file_sha256(context["repo"] / DIRECT_SOURCE),
            pad_source: file_sha256(context["repo"] / pad_source),
        },
        "profile_costs": {
            "font_normalization_s": font["duration_s"],
            "font_deepcopy_s": copy_cost,
            "main_recognition_s": recognition["duration_s"],
            "recognition_onnx_s": onnx["duration_s"],
        },
    }
    suspects = {
        "schema_version": SCHEMA_VERSION,
        "suspects": [
            {
                "id": "font-size-deepcopy",
                "route": "direct",
                "profile_pointer": {
                    "path": str(profile),
                    "line": copy_line,
                    "duration_s": copy_cost,
                },
                "source_pointer": {"path": DIRECT_SOURCE, "line": direct_line},
                "profile_source_establish": "A measured deepcopy call copies the input box list in font-size normalization.",
                "unknown": "Which nested objects must remain independent after normalization?",
                "question": "Could copying be narrowed while preserving downstream ownership?",
                "probe_decision": "A probe would address ownership safety; it is not required to locate this call.",
            },
            {
                "id": "recognition-bucket-widths",
                "route": "probe",
                "adapter": ADAPTER_NAME,
                "profile_pointer": {
                    "path": str(profile),
                    "line": recognition["line"],
                    "duration_s": recognition["duration_s"],
                },
                "source_pointers": [
                    {"path": SOURCE, "line": bucket_line},
                    {"path": SOURCE, "line": fallthrough_line},
                    {"path": SOURCE, "line": resize_line},
                    {"path": pad_source, "line": pad_line},
                ],
                "profile_source_establish": "Main recognition is costly; source routes crops to fixed width buckets and pads input.",
                "unknown": "Actual crop distribution, assigned widths, padding volume, and fallthrough frequency.",
                "question": "How much assigned recognition width is padding on this workload, and does original routing replay exactly?",
                "falsification": "No padding, or any mismatch between observed production widths and replayed routing.",
                "capture_boundary": "Main SSOCRCore.predict call after recognition crop creation.",
                "minimum_fields": "Crop width and height, effective height, width buckets, padding margin, padding flag.",
                "semantic_gate": "Original assigned bucket widths match geometry replay; complete raw page output matches adjacent controls.",
            },
        ],
    }
    atomic_json(run_dir / "intake.json", intake)
    atomic_json(run_dir / "suspects.json", suspects)
    atomic_json(
        run_dir / "probes" / "recognition-bucket-widths" / "plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "frozen_before_capture": True,
            **suspects["suspects"][1],
        },
    )
    atomic_json(
        run_dir / "reviewers" / "profile_only.json",
        {
            "schema_version": SCHEMA_VERSION,
            "frozen_before_capture": True,
            "inputs": ["profile", "profile_log", "inner_profile"],
            "established": [
                "main recognition and font-size normalization are profiled costs",
                "a deepcopy child is measured under font-size normalization",
            ],
            "unknown": [
                "copy ownership requirements",
                "recognition bucket implementation and crop distribution",
                "padding fraction and output equality under alternate widths",
            ],
        },
    )
    atomic_json(
        run_dir / "reviewers" / "profile_source.json",
        {
            "schema_version": SCHEMA_VERSION,
            "frozen_before_capture": True,
            "inputs": ["profile", "profile_log", "inner_profile", "subject_source"],
            "established": [
                "font-size-deepcopy call and profiled cost",
                "recognition fixed-width routing and padding mechanism",
                "last configured width is omitted from the routing comparison loop",
            ],
            "unknown": [
                "actual recognition crop distribution and padding denominator",
                "actual fallthrough count",
                "output equality under any alternative width policy",
            ],
        },
    )
    return intake


def _subject_command(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.diagnostic_enrichment.pilot",
        "subject-run",
        "--mode",
        mode,
        "--adapter",
        ADAPTER_NAME,
        "--iocr-repo",
        str(args.iocr_repo.resolve()),
        "--images-dir",
        str(args.images_dir.resolve()),
        "--run-dir",
        str(output),
        "--workload-sha256",
        args.workload_sha256,
        "--source-sha256",
        args.source_sha256,
    ]


def _full_page_gate(
    before: dict[str, Any], capture: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    sets = [
        [(page["image"], page["raw_output_sha256"]) for page in run["pages"]]
        for run in (before, capture, after)
    ]
    if any(len(pages) != 44 for pages in sets):
        raise ValueError("whole-page gate requires 44 pages per run")
    matched = sum(left == middle == right for left, middle, right in zip(*sets, strict=True))
    return {
        "passed": matched,
        "total": len(sets[0]),
        "mismatched_page_indices": [
            index for index, triple in enumerate(zip(*sets, strict=True)) if len(set(triple)) != 1
        ],
        "meaning": "complete raw OCR box digest matches control, capture, control in page order",
    }


def _report(run_dir: Path, intake: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    before, capture, after = (
        read_json(run_dir / directory / "result.json")
        for directory in ("baseline-before", "capture", "baseline-after")
    )
    manifest = read_json(run_dir / "capture" / "capture.json")
    manifest_digests = [entry["sha256"] for entry in manifest["full_raw_page_digests"]]
    capture_digests = [page["raw_output_sha256"] for page in capture["pages"]]
    if manifest_digests != capture_digests:
        raise ValueError("capture manifest full-page digests differ from subject result")
    full_gate = _full_page_gate(before, capture, after)
    local_gate = replay["local_semantic_gates"]
    totals = replay["totals"]
    gates_pass = (
        full_gate["passed"] == full_gate["total"] and local_gate["passed"] == local_gate["total"]
    )
    observed = totals["padding_width_units"] > 0
    probe_status = (
        "promoted" if gates_pass and observed else "rejected" if gates_pass else "inconclusive"
    )
    costs = intake["profile_costs"]
    suspects = read_json(run_dir / "suspects.json")["suspects"]
    direct = suspects[0]
    probed = suspects[1]
    recognition_source_pointers = list(probed["source_pointers"])
    resize_pointer = {
        "path": SOURCE,
        "line": _source_line(
            Path(intake["subject_root"]), SOURCE, "resized_w = math.ceil(imgH * ratio)"
        ),
    }
    if resize_pointer not in recognition_source_pointers:
        recognition_source_pointers.append(resize_pointer)
    candidates = {
        "schema_version": SCHEMA_VERSION,
        "subject_commit": PINNED_IOCR_COMMIT,
        "workload_sha256": intake["workload"]["sha256"],
        "counts": {
            "inspected": 2,
            "routed_direct": 1,
            "routed_probe": 1,
            "promoted": 1 + int(probe_status == "promoted"),
            "rejected": int(probe_status == "rejected"),
            "inconclusive": int(probe_status == "inconclusive"),
        },
        "findings": [
            {
                "id": direct["id"],
                "status": "promoted",
                "evidence_route": "direct",
                "evidence_level": "profile_plus_source",
                "profile_pointer": direct["profile_pointer"],
                "source_pointer": direct["source_pointer"],
                "observed_condition": "Deep copy is called in font-size normalization.",
                "profiled_inclusive_s": costs["font_deepcopy_s"],
                "denominator": {"profile_wall_s": intake["profile"]["wall_s"], "pages": 44},
                "correctness_gate": "not tested for a changed copy policy",
                "limitation": "Ownership requirements and avoidable cost are unknown; profile time is not a speedup estimate.",
                "next_check": "Capture object ownership and test a concrete narrower copy with full raw-output gates.",
            },
            {
                "id": probed["id"],
                "status": probe_status,
                "evidence_route": "probe_backed",
                "evidence_level": "runtime_geometry_replay"
                if gates_pass
                else "ungated_runtime_observation",
                "adapter": ADAPTER_NAME,
                "profile_pointer": probed["profile_pointer"],
                "source_pointers": recognition_source_pointers,
                "observed_condition": (
                    f"Captured geometry and the source resize rule imply "
                    f"{totals['padding_width_units']} of {totals['model_width_units']} assigned "
                    f"crop-width units ({replay['padding_fraction']:.2%}) were padding "
                    f"across {totals['crops']} crops."
                ),
                "denominator": {"crops": totals["crops"], "calls": totals["calls"], "pages": 44},
                "additional_counts": {
                    "fallthrough_crops": totals["fallthrough_crops"],
                    "last_bucket_crops": totals["last_bucket_crops"],
                    "above_last_width_crops": totals["above_last_width_crops"],
                },
                "correctness_gates": {"full_page": full_gate, "local_routing": local_gate},
                "negative_or_inconclusive": replay["negative_or_inconclusive"],
                "limitation": "Geometry replay validates original routing only. It cannot establish model-output equality or latency benefit for narrower buckets.",
                "next_check": "Replay a selected alternate-width model call with output equality gates, then use paired whole-page timing if a change is implemented.",
            },
        ],
    }
    atomic_json(run_dir / "candidates.json", candidates)
    atomic_json(
        run_dir / "direct" / "font-size-deepcopy" / "finding.json",
        candidates["findings"][0],
    )
    atomic_json(
        run_dir / "probes" / "recognition-bucket-widths" / "finding.json",
        candidates["findings"][1],
    )
    profile_only_path = run_dir / "reviewers" / "profile_only.json"
    profile_only = read_json(profile_only_path) if profile_only_path.exists() else None
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "profile_only_review": "reviewers/profile_only.json" if profile_only else None,
        "profile_only_review_sha256": file_sha256(profile_only_path) if profile_only else None,
        "profile_only_frozen_before_capture": (
            bool(profile_only["frozen_before_capture"]) if profile_only else False
        ),
        "frozen_profile_source_review": "reviewers/profile_source.json",
        "frozen_profile_source_sha256": file_sha256(run_dir / "reviewers" / "profile_source.json"),
        "frozen_suspects_sha256": file_sha256(run_dir / "suspects.json"),
        "enriched_review": "reviewers/enriched.json",
        "new_condition_relative_to_profile_source": (
            [
                "Measured crop distribution and padding fraction under the effective workload configuration"
            ]
            if gates_pass and observed
            else []
        ),
        "confirmation_of_source_visible_mechanism": (
            [
                "The configured last bucket received no crops; observed fallthrough frequency is recorded separately"
            ]
            if gates_pass
            else []
        ),
        "new_conditions_count": int(gates_pass and observed),
        "source_visible_confirmations_count": int(gates_pass),
        "new_project_findings_count": 0,
        "prior_work_limit": "The standalone recognition-bucket PoC already measured this workload; this run validates an integrated contract, not a previously unknown project finding.",
        "reviewer_limit": (
            "Profile-plus-source claims were frozen before capture. No independent blinded "
            "human or agent reviewer was run; the profile-only snapshot was reconstructed "
            "after capture for this first execution."
            if profile_only and not profile_only["frozen_before_capture"]
            else "Reviewer claims were frozen before capture, but no independent blinded "
            "human or agent reviewer was run."
        ),
    }
    atomic_json(run_dir / "evaluation.json", evaluation)
    atomic_json(
        run_dir / "reviewers" / "enriched.json",
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": ["profile", "profile_log", "inner_profile", "subject_source", "gated_replay"],
            "established": [
                {
                    "condition": "measured recognition padding fraction",
                    "count": totals["padding_width_units"],
                    "denominator": totals["model_width_units"],
                    "gate_passed": gates_pass,
                    "evidence": "replay.json",
                },
                {
                    "condition": "observed source-visible last-bucket fallthrough",
                    "count": totals["fallthrough_crops"],
                    "denominator": totals["crops"],
                    "gate_passed": gates_pass,
                    "evidence": "replay.json",
                },
            ],
            "unknown": ["model-output equality and page latency under a changed bucket policy"],
        },
    )
    elapsed = [run["elapsed_s"] for run in (before, capture, after)]
    midpoint = (elapsed[0] + elapsed[2]) / 2
    do_predict = [
        sum(page["do_predict_ns"] for page in run["pages"]) / 1e9
        for run in (before, capture, after)
    ]
    timing = {
        "whole_run_s": elapsed,
        "whole_run_capture_vs_control_midpoint_s": elapsed[1] - midpoint,
        "whole_run_capture_vs_control_midpoint_pct": 100 * (elapsed[1] / midpoint - 1),
        "control_drift_s": elapsed[2] - elapsed[0],
        "do_predict_s": do_predict,
        "replay_validation_s": replay["validation_s"],
        "replay_diagnostic_s": replay["replay_s"],
    }
    atomic_json(run_dir / "timing.json", timing)
    fresh_profile_path = run_dir / "fresh_profile_summary.json"
    fresh_profile = read_json(fresh_profile_path) if fresh_profile_path.exists() else None
    report_lines = [
        "# Profiler-guided diagnostic enrichment pilot",
        "",
        f"IOCR `{PINNED_IOCR_COMMIT}`; 44 English/CPU pages; workload SHA-256 `{intake['workload']['sha256']}`.",
        "",
        "## Evidence and routing",
        "",
        f"The pinned input profile took {intake['profile']['wall_s']:.3f} s. Main recognition was {costs['main_recognition_s']:.3f} s inclusive, including {costs['recognition_onnx_s']:.3f} s in ONNX. Font-size normalization was {costs['font_normalization_s']:.3f} s, with {costs['font_deepcopy_s']:.3f} s in its direct `deepcopy` child.",
        f"The direct candidate points to profile line {direct['profile_pointer']['line']} and `{DIRECT_SOURCE}:{direct['source_pointer']['line']}`. It is a copying investigation; safe removal was not tested.",
        f"The probe candidate points to profile line {probed['profile_pointer']['line']}, `{SOURCE}:{probed['source_pointers'][0]['line']}` and `{SOURCE}:{probed['source_pointers'][1]['line']}`. The source shows fixed buckets and a last-bucket fallthrough, but not actual crop frequencies.",
        "",
        "## Fresh capture and replay",
        "",
        f"Whole-run control, capture, control: {elapsed[0]:.3f}, {elapsed[1]:.3f}, {elapsed[2]:.3f} s. Capture was {timing['whole_run_capture_vs_control_midpoint_s']:+.3f} s ({timing['whole_run_capture_vs_control_midpoint_pct']:+.2f}%) versus the control midpoint; control drift was {timing['control_drift_s']:+.3f} s. This single-bracket difference estimates capture overhead and is not an optimization result.",
        f"The capture wrote {replay['validated_capsules']} local geometry capsules totaling {read_json(run_dir / 'capture' / 'capture.json')['total_capsule_bytes']} bytes. Validation took {replay['validation_s']:.3f} s; diagnostic replay took {replay['replay_s']:.3f} s.",
        f"Complete raw page digests matched {full_gate['passed']}/{full_gate['total']}; original bucket assignments matched geometry replay {local_gate['passed']}/{local_gate['total']} calls. Mismatched page indices: {full_gate['mismatched_page_indices']}.",
        f"From captured geometry and the source resize rule at `{SOURCE}:{resize_pointer['line']}`, across {totals['crops']} crops in {totals['calls']} calls, padding was {totals['padding_width_units']}/{totals['model_width_units']} assigned width units ({replay['padding_fraction']:.2%}); fallthrough count was {totals['fallthrough_crops']} and last-bucket occupancy was {totals['last_bucket_crops']}. These are width units, not CPU seconds.",
        f"Probe outcome: **{probe_status}**. Above-maximum crops: {totals['above_last_width_crops']}/{totals['crops']} (negative when zero). Alternate-width output equality is inconclusive because geometry capsules cannot replay model inference.",
        "",
        "## Novelty and limits",
        "",
        "The measured crop distribution and padding fraction were unavailable to the frozen profile-plus-source review. The fallthrough mechanism and unused last-bucket path are source-visible; this run measures their workload frequency. Count one new runtime condition relative to that review and one source-visible confirmation. The earlier standalone PoC had already measured this workload, so there are zero new project findings. No independent blinded reviewer ran.",
        "No IOCR production source was changed. A narrower bucket policy has no model-output equality gate here, and no page-latency speedup is claimed.",
        "",
        "Machine-readable artifacts: `intake.json`, `suspects.json`, `capture/capture.json`, `replay.json`, `candidates.json`, `evaluation.json`, `timing.json`, `status.json`. Capsules and run logs remain local and are not included in this report.",
        "",
    ]
    if fresh_profile:
        measured = fresh_profile["profile_costs"]
        report_lines[report_lines.index("## Fresh capture and replay")] = (
            "## Fresh profiler cross-check\n\n"
            f"An independent post-bracket profiler pass took {fresh_profile['profile_wall_s']:.3f} s "
            f"inside inference ({fresh_profile['whole_run_s']:.3f} s whole run). "
            f"It attributed {measured['main_recognition_s']:.3f} s to main recognition, "
            f"{measured['recognition_onnx_s']:.3f} s to its ONNX child, and "
            f"{measured['font_deepcopy_s']:.3f} s to the font-size `deepcopy`. "
            "Profile costs are inclusive and varied between passes.\n\n"
            "## Fresh capture and replay"
        )
    atomic_bytes(run_dir / "report.md", "\n".join(report_lines).encode("utf-8"))
    return candidates


def _append_step_times(run_dir: Path, status: dict[str, Any]) -> None:
    lines = ["## Pilot wall time by step", ""]
    for name in (
        "intake_s",
        "baseline-before_s",
        "capture_s",
        "baseline-after_s",
        "replay_s",
        "report_s",
    ):
        if name in status["steps"]:
            lines.append(f"- {name}: {status['steps'][name]:.3f} s")
    lines.extend([f"- total: {status['total_s']:.3f} s", ""])
    original = (run_dir / "report.md").read_bytes()
    atomic_bytes(run_dir / "report.md", original + b"\n" + "\n".join(lines).encode("utf-8"))


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite pilot run {run_dir}")
    started = time.perf_counter_ns()
    context = _preflight(args)
    run_dir.mkdir(parents=True)
    status: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "state": "running", "steps": {}}
    atomic_json(run_dir / "status.json", status)
    try:
        intake_started = time.perf_counter_ns()
        intake = _intake(args, context, run_dir)
        status["steps"]["intake_s"] = (time.perf_counter_ns() - intake_started) / 1e9
        atomic_json(run_dir / "status.json", status)
        args.workload_sha256 = context["workload_sha256"]
        args.source_sha256 = context["source_sha256"]
        for label, mode in (
            ("baseline-before", "baseline"),
            ("capture", "capture"),
            ("baseline-after", "baseline"),
        ):
            step_started = time.perf_counter_ns()
            print(f"starting {label}", flush=True)
            subprocess.run(
                _subject_command(args, mode, run_dir / label),
                check=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            status["steps"][f"{label}_s"] = (time.perf_counter_ns() - step_started) / 1e9
            atomic_json(run_dir / "status.json", status)
        replay_started = time.perf_counter_ns()
        replay = ADAPTERS[ADAPTER_NAME]["replay"](
            capture_dir=run_dir / "capture",
            subject_commit=PINNED_IOCR_COMMIT,
            workload_sha256=context["workload_sha256"],
            source_sha256=context["source_sha256"],
        )
        atomic_json(run_dir / "replay.json", replay)
        status["steps"]["replay_s"] = (time.perf_counter_ns() - replay_started) / 1e9
        atomic_json(run_dir / "status.json", status)
        report_started = time.perf_counter_ns()
        candidates = _report(run_dir, intake, replay)
        status["steps"]["report_s"] = (time.perf_counter_ns() - report_started) / 1e9
        status["state"] = "done"
        status["total_s"] = (time.perf_counter_ns() - started) / 1e9
        atomic_json(run_dir / "status.json", status)
        _append_step_times(run_dir, status)
        print(f"pilot complete: {candidates['counts']} -> {run_dir / 'report.md'}", flush=True)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["total_s"] = (time.perf_counter_ns() - started) / 1e9
        atomic_json(run_dir / "status.json", status)
        raise


def subject_run(args: argparse.Namespace) -> None:
    ADAPTERS[args.adapter]["capture"](
        mode=args.mode,
        repo=args.iocr_repo.resolve(),
        images_dir=args.images_dir.resolve(),
        run_dir=args.run_dir.resolve(),
        workload_sha256=args.workload_sha256,
        source_sha256=args.source_sha256,
        subject_commit=PINNED_IOCR_COMMIT,
    )


def replay_only(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    intake = read_json(run_dir / "intake.json")
    manifest = read_json(run_dir / "capture" / "capture.json")
    adapter = manifest["adapter"]
    if adapter not in ADAPTERS:
        raise ValueError(f"unregistered adapter {adapter}")
    result = ADAPTERS[adapter]["replay"](
        capture_dir=run_dir / "capture",
        subject_commit=intake["subject_commit"],
        workload_sha256=intake["workload"]["sha256"],
        source_sha256=intake["source"][SOURCE],
    )
    atomic_json(run_dir / "replay.json", result)
    print(f"validated {result['validated_capsules']} capsules")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--iocr-repo", type=Path, required=True)
    run_parser.add_argument("--images-dir", type=Path, required=True)
    run_parser.add_argument("--profile", type=Path, required=True)
    run_parser.add_argument("--profile-log", type=Path, required=True)
    run_parser.add_argument("--inner-profile", type=Path, required=True)
    run_parser.add_argument("--profile-manifest", type=Path, required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.set_defaults(action=run)
    subject_parser = commands.add_parser("subject-run", help=argparse.SUPPRESS)
    subject_parser.add_argument("--mode", choices=("baseline", "capture"), required=True)
    subject_parser.add_argument("--adapter", choices=tuple(ADAPTERS), required=True)
    subject_parser.add_argument("--iocr-repo", type=Path, required=True)
    subject_parser.add_argument("--images-dir", type=Path, required=True)
    subject_parser.add_argument("--run-dir", type=Path, required=True)
    subject_parser.add_argument("--workload-sha256", required=True)
    subject_parser.add_argument("--source-sha256", required=True)
    subject_parser.set_defaults(action=subject_run)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--run-dir", type=Path, required=True)
    replay_parser.set_defaults(action=replay_only)
    arguments = parser.parse_args()
    arguments.action(arguments)


if __name__ == "__main__":
    main()
