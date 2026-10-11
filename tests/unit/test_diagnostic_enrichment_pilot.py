"""Focused integrity and semantic-gate tests for the standalone pilot."""

from __future__ import annotations

import pytest

from experiments.diagnostic_enrichment.contract import (
    MAX_CAPSULE_BYTES,
    MAX_TOTAL_BYTES,
    PRIVACY_CLASS,
    SCHEMA_VERSION,
    CapsuleWriter,
    atomic_json,
    object_sha256,
    read_json,
)
from experiments.diagnostic_enrichment.pilot import _full_page_gate
from experiments.diagnostic_enrichment.recognition_bucket import (
    ADAPTER_NAME,
    ADAPTER_VERSION,
    replay,
)


def _small_capture(tmp_path):
    root = tmp_path / "capture"
    config = {"img_h": 32, "widths": [100, 200, 300], "padding_margin": 1.0, "pad_enabled": True}
    writer = CapsuleWriter(root, adapter=ADAPTER_NAME, adapter_version=ADAPTER_VERSION)
    writer.add(
        page_index=0,
        call_index=0,
        capsule={
            "schema_version": SCHEMA_VERSION,
            "adapter": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "page_index": 0,
            "call_index": 0,
            "config": config,
            "crop_sizes": [[50, 32], [200, 32]],
        },
        # These are the widths the source loop would actually send to predict_batch.
        captured_semantic_sha256=object_sha256([100, 200]),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adapter": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "subject_commit": "commit",
        "workload_sha256": "workload",
        "source_sha256": "source",
        "config_sha256": object_sha256(config),
        "effective_config": config,
        "privacy_class": PRIVACY_CLASS,
        "retention": "local_only",
        "max_capsule_bytes": MAX_CAPSULE_BYTES,
        "max_total_bytes": MAX_TOTAL_BYTES,
        "capsule_count": len(writer.records),
        "total_capsule_bytes": writer.total_bytes,
        "records": writer.records,
    }
    atomic_json(root / "capture.json", manifest)
    return root


def test_replay_uses_observed_source_assignments_as_semantic_gate(tmp_path):
    root = _small_capture(tmp_path)
    report = replay(
        capture_dir=root,
        subject_commit="commit",
        workload_sha256="workload",
        source_sha256="source",
    )
    assert report["local_semantic_gates"]["passed"] == 1
    assert report["totals"]["crops"] == 2
    assert report["totals"]["padding_width_units"] == 50
    assert report["totals"]["fallthrough_crops"] == 1
    assert report["totals"]["last_bucket_crops"] == 0
    assert report["negative_or_inconclusive"][0]["outcome"] == "negative"
    assert report["negative_or_inconclusive"][1]["outcome"] == "inconclusive"


def test_replay_rejects_tampered_capsule_before_diagnosis(tmp_path):
    root = _small_capture(tmp_path)
    capsule = next((root / "capsules").glob("*.json"))
    original = capsule.read_bytes()
    capsule.write_bytes(original.replace(b"[50,32]", b"[60,32]"))
    with pytest.raises(ValueError, match="checksum"):
        replay(
            capture_dir=root,
            subject_commit="commit",
            workload_sha256="workload",
            source_sha256="source",
        )


def test_replay_rejects_wrong_workload_and_path_escape(tmp_path):
    root = _small_capture(tmp_path)
    with pytest.raises(ValueError, match="workload_sha256"):
        replay(
            capture_dir=root,
            subject_commit="commit",
            workload_sha256="different-workload",
            source_sha256="source",
        )
    manifest = read_json(root / "capture.json")
    manifest["records"][0]["capsule"] = "../other.json"
    atomic_json(root / "capture.json", manifest)
    with pytest.raises(ValueError, match="escapes"):
        replay(
            capture_dir=root,
            subject_commit="commit",
            workload_sha256="workload",
            source_sha256="source",
        )


def test_capture_refuses_capsule_above_byte_budget(tmp_path):
    writer = CapsuleWriter(tmp_path, adapter=ADAPTER_NAME, adapter_version=ADAPTER_VERSION)
    with pytest.raises(ValueError, match="capsule exceeds"):
        writer.add(
            page_index=0,
            call_index=0,
            capsule={
                "schema_version": SCHEMA_VERSION,
                "adapter": ADAPTER_NAME,
                "adapter_version": ADAPTER_VERSION,
                "large": "x" * MAX_CAPSULE_BYTES,
            },
            captured_semantic_sha256=object_sha256([]),
        )
    assert not (tmp_path / "capsules").exists()


def test_full_page_gate_reports_exact_mismatch(tmp_path):
    del tmp_path
    pages = [{"image": f"page-{i}", "raw_output_sha256": f"digest-{i}"} for i in range(44)]
    before = {"pages": pages}
    capture = {"pages": [dict(page) for page in pages]}
    after = {"pages": [dict(page) for page in pages]}
    capture["pages"][7]["raw_output_sha256"] = "changed"
    gate = _full_page_gate(before, capture, after)
    assert gate["passed"] == 43
    assert gate["mismatched_page_indices"] == [7]
