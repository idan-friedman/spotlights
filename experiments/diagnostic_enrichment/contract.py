"""Versioned, privacy-bounded capsule contract shared by probe adapters."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_CAPSULE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
PRIVACY_CLASS = "local_geometry_sensitive"


def encoded_json(value: Any) -> bytes:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (rendered + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return sha256_bytes(encoded_json(value))


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, encoded_json(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class CapsuleWriter:
    """Enforce limits before writing each complete invocation capsule."""

    def __init__(self, root: Path, *, adapter: str, adapter_version: int) -> None:
        self.root = root
        self.adapter = adapter
        self.adapter_version = adapter_version
        self.records: list[dict[str, Any]] = []
        self.total_bytes = 0

    def add(
        self,
        *,
        page_index: int,
        call_index: int,
        capsule: dict[str, Any],
        captured_semantic_sha256: str,
    ) -> None:
        if capsule.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("capsule schema version differs")
        if capsule.get("adapter") != self.adapter:
            raise ValueError("capsule adapter differs")
        if capsule.get("adapter_version") != self.adapter_version:
            raise ValueError("capsule adapter version differs")
        data = encoded_json(capsule)
        if len(data) > MAX_CAPSULE_BYTES:
            raise ValueError(f"capsule exceeds {MAX_CAPSULE_BYTES} bytes")
        if self.total_bytes + len(data) > MAX_TOTAL_BYTES:
            raise ValueError(f"capsules exceed {MAX_TOTAL_BYTES} total bytes")
        relative = f"capsules/page-{page_index:03d}-call-{call_index:03d}.json"
        path = self.root / relative
        if path.exists():
            raise FileExistsError(path)
        atomic_bytes(path, data)
        self.records.append(
            {
                "page_index": page_index,
                "call_index": call_index,
                "capsule": relative,
                "capsule_bytes": len(data),
                "capsule_sha256": sha256_bytes(data),
                "captured_semantic_sha256": captured_semantic_sha256,
            }
        )
        self.total_bytes += len(data)


def validate_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    adapter: str,
    adapter_version: int,
    subject_commit: str,
    workload_sha256: str,
    source_sha256: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Validate every capsule before any diagnostic replay starts."""
    expected = {
        "schema_version": SCHEMA_VERSION,
        "adapter": adapter,
        "adapter_version": adapter_version,
        "subject_commit": subject_commit,
        "workload_sha256": workload_sha256,
        "source_sha256": source_sha256,
        "privacy_class": PRIVACY_CLASS,
        "max_capsule_bytes": MAX_CAPSULE_BYTES,
        "max_total_bytes": MAX_TOTAL_BYTES,
        "retention": "local_only",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} differs")
    if object_sha256(manifest.get("effective_config")) != manifest.get("config_sha256"):
        raise ValueError("manifest effective configuration hash differs")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest has no invocation records")
    if manifest.get("capsule_count") != len(records):
        raise ValueError("manifest capsule count differs")
    seen: set[tuple[int, int]] = set()
    seen_paths: set[Path] = set()
    total = 0
    checked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    capsule_root = (root / "capsules").resolve()
    for record in records:
        key = (record["page_index"], record["call_index"])
        if key in seen:
            raise ValueError("duplicate capsule invocation")
        seen.add(key)
        relative = Path(record["capsule"])
        path = (root / relative).resolve()
        if not path.is_relative_to(capsule_root) or path.suffix != ".json":
            raise ValueError("capsule path escapes local capsule directory")
        if path in seen_paths:
            raise ValueError("duplicate capsule path")
        seen_paths.add(path)
        size = path.stat().st_size
        if size != record["capsule_bytes"] or size > MAX_CAPSULE_BYTES:
            raise ValueError("capsule byte count or limit differs")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("total capsule byte limit exceeded")
        data = path.read_bytes()
        if sha256_bytes(data) != record["capsule_sha256"]:
            raise ValueError("capsule checksum differs")
        capsule = json.loads(data)
        if capsule.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("capsule schema differs")
        if capsule.get("adapter") != adapter or capsule.get("adapter_version") != adapter_version:
            raise ValueError("capsule adapter differs")
        if capsule.get("page_index") != key[0] or capsule.get("call_index") != key[1]:
            raise ValueError("capsule invocation differs")
        if object_sha256(capsule["config"]) != manifest["config_sha256"]:
            raise ValueError("capsule configuration differs")
        checked.append((record, capsule))
    if total != manifest.get("total_capsule_bytes"):
        raise ValueError("manifest total capsule bytes differs")
    actual_files = {path.resolve() for path in (root / "capsules").glob("*.json")}
    expected_files = {(root / record["capsule"]).resolve() for record in records}
    if actual_files != expected_files:
        raise ValueError("unlisted or missing capsule file")
    return checked
