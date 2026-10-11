"""Adapt the existing recognition-bucket probe to small, text-free capsules."""

# ruff: noqa: E501

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from experiments.diagnostic_enrichment.contract import (
    MAX_CAPSULE_BYTES,
    MAX_TOTAL_BYTES,
    PRIVACY_CLASS,
    SCHEMA_VERSION,
    CapsuleWriter,
    atomic_json,
    file_sha256,
    object_sha256,
    validate_manifest,
)
from experiments.diagnostic_enrichment.probe_helpers import _digest, _routing, _tight_widths

ADAPTER_NAME = "recognition_bucket_geometry"
ADAPTER_VERSION = 1
SOURCE = "hrl_ocr/models/recognition/ssocr/ssocr_core.py"


def _configuration(core: Any, height: int, widths: list[int]) -> dict[str, Any]:
    return {
        "img_h": int(height),
        "widths": [int(width) for width in widths],
        "padding_margin": float(core.padding_margin),
        "pad_enabled": bool(core.opt.PAD),
    }


def run_subject(
    *,
    mode: str,
    repo: Path,
    images_dir: Path,
    run_dir: Path,
    workload_sha256: str,
    source_sha256: str,
    subject_commit: str,
) -> dict[str, Any]:
    """Run the original whole-page path; capture only selected recognition geometry."""
    if mode not in ("baseline", "capture"):
        raise ValueError("unknown run mode")
    if run_dir.is_relative_to(repo) or run_dir.is_relative_to(images_dir):
        raise ValueError("run artifacts must be outside the IOCR checkout and image workload")
    if not os.environ.get("IOCR_WEIGHTS_LOCATION"):
        raise ValueError("IOCR_WEIGHTS_LOCATION is required")
    os.environ["IOCR_WEIGHTS_LOCATION"] = str(Path(os.environ["IOCR_WEIGHTS_LOCATION"]).resolve())
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if head != subject_commit or dirty:
        raise ValueError("subject commit or clean-checkout gate failed")
    if file_sha256(repo / SOURCE) != source_sha256:
        raise ValueError("subject recognition source hash differs")
    images = sorted(images_dir.glob("*.jpg"))
    workload = [
        {"name": image.name, "bytes": image.stat().st_size, "sha256": file_sha256(image)}
        for image in images
    ]
    if object_sha256(workload) != workload_sha256:
        raise ValueError("subject workload hash differs")
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    sys.dont_write_bytecode = True
    os.environ["USE_CPU_ONLY"] = "True"
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from hrl_ocr.common_utils import convert_output
    from hrl_ocr.common_utils.dump_data import DumpData
    from hrl_ocr.models.recognition.ssocr.ssocr_core import SSOCRCore
    from hrl_ocr.pipeline.glue_components.recognizer_by_script import RecognizerByScript
    from hrl_ocr.pipeline.ibm_ocr.ocr_inference import OCRInference
    from hrl_ocr.tests.ocr.ocr_folder_processor import OCRFolderProcessor

    writer = CapsuleWriter(run_dir, adapter=ADAPTER_NAME, adapter_version=ADAPTER_VERSION)
    pages: list[dict[str, Any]] = []
    current_page: int | None = None
    in_main_recognition = False
    config: dict[str, Any] | None = None
    original_single_image = OCRFolderProcessor._OCRFolderProcessor__run_single_image
    original_predict = OCRInference.do_predict
    original_recognize = RecognizerByScript.recognize_according_to_script
    original_ssocr_predict = SSOCRCore.predict

    @wraps(original_single_image)
    def measured_single_image(
        processor: Any,
        inference: Any,
        pil_image: Any,
        image_base_name: str,
        image_extension: str,
        *arguments: Any,
        **keywords: Any,
    ) -> Any:
        nonlocal current_page
        if current_page is not None:
            raise ValueError("nested image processing")
        current_page = len(pages)
        pages.append({"index": current_page, "image": image_base_name + image_extension})
        try:
            result = original_single_image(
                processor,
                inference,
                pil_image,
                image_base_name,
                image_extension,
                *arguments,
                **keywords,
            )
            if "raw_output_sha256" not in pages[-1]:
                raise ValueError("page has no complete raw OCR digest")
            print(f"{mode}: completed page {len(pages)}", file=sys.__stdout__, flush=True)
            return result
        finally:
            current_page = None

    @wraps(original_predict)
    def measured_predict(inference: Any, *arguments: Any, **keywords: Any) -> Any:
        started = time.perf_counter_ns()
        result = original_predict(inference, *arguments, **keywords)
        if current_page is None:
            raise ValueError("do_predict outside tracked page")
        pages[current_page]["do_predict_ns"] = time.perf_counter_ns() - started
        pages[current_page]["raw_output_sha256"] = _digest(result[0])
        return result

    @wraps(original_recognize)
    def measured_recognize(recognizer: Any, *arguments: Any, **keywords: Any) -> Any:
        nonlocal in_main_recognition
        if in_main_recognition:
            raise ValueError("nested main recognition")
        in_main_recognition = True
        try:
            return original_recognize(recognizer, *arguments, **keywords)
        finally:
            in_main_recognition = False

    @wraps(original_ssocr_predict)
    def measured_ssocr_predict(
        core: Any,
        images: list[Any],
        imgH: int = 32,
        imgW: Any = None,
        profiler: Any = None,
    ) -> Any:
        nonlocal config
        if imgW is None:
            imgW = core.opt.imgW
        if mode == "baseline" or not in_main_recognition or current_page is None:
            return original_ssocr_predict(core, images, imgH, imgW, profiler)
        if not isinstance(imgW, list) or len(imgW) < 2 or not core.opt.PAD:
            raise ValueError("recognition adapter requires padded width buckets")
        call_config = _configuration(core, imgH, imgW)
        if config is None:
            config = call_config
        elif config != call_config:
            raise ValueError("effective recognition configuration changed during workload")
        sizes = [[int(width), int(height)] for width, height in (image.size for image in images)]
        positions: dict[int, list[int]] = defaultdict(list)
        for index, image in enumerate(images):
            positions[id(image)].append(index)
        assigned: list[int | None] = [None] * len(images)
        original_batch = core.predict_batch

        def observed_batch(
            batch_images: list[Any], height: int, width: int, batch_profiler: Any
        ) -> Any:
            for image in batch_images:
                indices = positions[id(image)]
                if not indices:
                    raise ValueError("recognition batch contains an unknown crop")
                index = indices.pop(0)
                if assigned[index] is not None:
                    raise ValueError("recognition crop assigned twice")
                assigned[index] = int(width)
            return original_batch(batch_images, height, width, batch_profiler)

        with patch.object(core, "predict_batch", observed_batch):
            result = original_ssocr_predict(core, images, imgH, imgW, profiler)
        if any(value is None for value in assigned):
            raise ValueError("recognition crop was not assigned a bucket")
        if len(result) != len(images):
            raise ValueError("recognition output count differs from input count")
        call_index = sum(record["page_index"] == current_page for record in writer.records)
        capsule = {
            "schema_version": SCHEMA_VERSION,
            "adapter": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "page_index": current_page,
            "call_index": call_index,
            "config": call_config,
            "crop_sizes": sizes,
        }
        writer.add(
            page_index=current_page,
            call_index=call_index,
            capsule=capsule,
            captured_semantic_sha256=object_sha256(assigned),
        )
        return result

    started = time.perf_counter_ns()
    with (
        patch.object(convert_output, "convert_to_WebUI", lambda *a, **k: {}),
        patch.object(DumpData, "dump_output_json", lambda *a, **k: None),
        patch.object(
            OCRFolderProcessor, "_OCRFolderProcessor__run_single_image", measured_single_image
        ),
        patch.object(OCRInference, "do_predict", measured_predict),
        patch.object(RecognizerByScript, "recognize_according_to_script", measured_recognize),
        patch.object(SSOCRCore, "predict", measured_ssocr_predict),
        (run_dir / "run.log").open("w", encoding="utf-8") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        OCRFolderProcessor().process_folder(
            folder_of_images=str(images_dir),
            output_location=str(run_dir / "output"),
            languages_list=["eng"],
            visualize_results=False,
            dump_debug_images=False,
        )
    elapsed_s = (time.perf_counter_ns() - started) / 1e9
    if len(pages) != len(list(images_dir.glob("*.jpg"))):
        raise ValueError("whole workload was not processed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "subject_commit": subject_commit,
        "workload_sha256": workload_sha256,
        "elapsed_s": elapsed_s,
        "pages": pages,
    }
    atomic_json(run_dir / "result.json", result)
    if mode == "capture":
        if config is None or not writer.records:
            raise ValueError("selected recognition boundary did not run")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "adapter": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "subject_commit": subject_commit,
            "workload_sha256": workload_sha256,
            "source_sha256": source_sha256,
            "config_sha256": object_sha256(config),
            "effective_config": config,
            "privacy_class": PRIVACY_CLASS,
            "retention": "local_only",
            "max_capsule_bytes": MAX_CAPSULE_BYTES,
            "max_total_bytes": MAX_TOTAL_BYTES,
            "capsule_count": len(writer.records),
            "total_capsule_bytes": writer.total_bytes,
            "full_raw_page_digests": [
                {"page_index": page["index"], "sha256": page["raw_output_sha256"]} for page in pages
            ],
            "records": writer.records,
        }
        atomic_json(run_dir / "capture.json", manifest)
    return result


def replay(
    *,
    capture_dir: Path,
    subject_commit: str,
    workload_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Validate all capsules, replay the existing routing helper, then diagnose."""
    started = time.perf_counter_ns()
    from experiments.diagnostic_enrichment.contract import read_json

    manifest = read_json(capture_dir / "capture.json")
    checked = validate_manifest(
        capture_dir,
        manifest,
        adapter=ADAPTER_NAME,
        adapter_version=ADAPTER_VERSION,
        subject_commit=subject_commit,
        workload_sha256=workload_sha256,
        source_sha256=source_sha256,
    )
    validation_s = (time.perf_counter_ns() - started) / 1e9
    replay_started = time.perf_counter_ns()
    events: list[dict[str, Any]] = []
    totals = {
        "calls": 0,
        "crops": 0,
        "model_width_units": 0,
        "resized_width_units": 0,
        "padding_width_units": 0,
        "fallthrough_crops": 0,
        "above_last_width_crops": 0,
        "last_bucket_crops": 0,
        "tightenable_calls": 0,
    }
    for record, capsule in checked:
        config = capsule["config"]
        if not config["pad_enabled"]:
            raise ValueError("recognition adapter requires padding enabled")
        sizes = capsule["crop_sizes"]
        if not isinstance(sizes, list) or not sizes:
            raise ValueError("empty recognition invocation")
        if any(
            not isinstance(size, list)
            or len(size) != 2
            or any(not isinstance(value, int) or value <= 0 for value in size)
            for size in sizes
        ):
            raise ValueError("invalid crop geometry")
        crops = [SimpleNamespace(size=tuple(size)) for size in sizes]
        widths = config["widths"]
        routing = _routing(crops, config["img_h"], widths, config["padding_margin"])
        assigned = [widths[int(row["bucket_index"])] for row in routing]
        gate = object_sha256(assigned) == record["captured_semantic_sha256"]
        tight = _tight_widths(routing, widths)
        padded = sum(assigned)
        resized = sum(int(row["resized_width"]) for row in routing)
        event = {
            "page_index": record["page_index"],
            "call_index": record["call_index"],
            "local_routing_gate": gate,
            "crops": len(sizes),
            "model_width_units": padded,
            "resized_width_units": resized,
            "padding_width_units": padded - resized,
            "fallthrough_crops": sum(bool(row["fallthrough"]) for row in routing),
            "above_last_width_crops": sum(bool(row["above_last_width"]) for row in routing),
            "last_bucket_crops": sum(value == widths[-1] for value in assigned),
            "tightenable": tight != widths,
        }
        events.append(event)
        totals["calls"] += 1
        for key in (
            "crops",
            "model_width_units",
            "resized_width_units",
            "padding_width_units",
            "fallthrough_crops",
            "above_last_width_crops",
            "last_bucket_crops",
        ):
            totals[key] += event[key]
        totals["tightenable_calls"] += int(event["tightenable"])
    replay_s = (time.perf_counter_ns() - replay_started) / 1e9
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter": ADAPTER_NAME,
        "validated_capsules": len(checked),
        "validation_s": validation_s,
        "replay_s": replay_s,
        "local_semantic_gates": {
            "passed": sum(event["local_routing_gate"] for event in events),
            "total": len(events),
            "meaning": "replayed geometry routing equals widths observed in the original IOCR call",
        },
        "totals": totals,
        "padding_fraction": totals["padding_width_units"] / totals["model_width_units"],
        "events": events,
        "negative_or_inconclusive": [
            {
                "question": "Are any crops wider than the largest configured bucket?",
                "outcome": "negative" if totals["above_last_width_crops"] == 0 else "positive",
                "count": totals["above_last_width_crops"],
                "denominator": totals["crops"],
            },
            {
                "question": "Would narrower buckets preserve complete recognized output?",
                "outcome": "inconclusive",
                "reason": "Geometry-only capsules cannot replay model inference or recognized text.",
            },
        ],
    }
