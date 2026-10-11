"""Local copies of the helpers used by the earlier IOCR probe PoCs.

Routing and width selection come from profiler_recognition_bucket_probe/poc.py.
The canonical output digest comes from targeted_font_size_probe/poc.py.
Keeping these exact helpers here makes the pilot runnable from a clean branch.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if all(hasattr(value, name) for name in ("detach", "cpu", "tolist")):
        return _canonical(value.detach().cpu().tolist())
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return _canonical(value.tolist())
    if hasattr(value, "item"):
        return _canonical(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _routing(
    crops: list[Any], img_h: int, widths: list[int], margin: float
) -> list[dict[str, int | float | bool]]:
    if len(widths) < 2 or sorted(set(widths)) != widths:
        raise ValueError("recognition bucket widths must be strictly increasing")
    if img_h <= 0 or margin <= 0:
        raise ValueError("invalid recognition height or padding margin")
    routed = []
    for crop in crops:
        width, height = crop.size
        if width <= 0 or height <= 0:
            raise ValueError("invalid crop dimensions")
        scaled = img_h * width / height
        required = scaled * margin
        bucket_index = next(
            (index for index, limit in enumerate(widths[:-1]) if required < limit),
            len(widths) - 2,
        )
        assigned_width = widths[bucket_index]
        resized_width = min(assigned_width, math.ceil(scaled))
        routed.append(
            {
                "bucket_index": bucket_index,
                "required_width": required,
                "scaled_width": scaled,
                "resized_width": resized_width,
                "fallthrough": required >= widths[-2],
                # Legacy schema name: the whole image is compressed to the
                # bucket width by transform(); no pixels are cut off.
                "clipped": math.ceil(scaled) > assigned_width,
                "above_last_width": math.ceil(scaled) > widths[-1],
            }
        )
    return routed


def _tight_widths(
    routing: list[dict[str, int | float | bool]], widths: list[int], align: int = 16
) -> list[int]:
    if align <= 0:
        raise ValueError("alignment must be positive")
    alternative = list(widths)
    for index in range(len(widths) - 1):
        members = [entry for entry in routing if entry["bucket_index"] == index]
        if not members or any(entry["fallthrough"] for entry in members):
            continue
        # Routing uses strict '<'. The transformed image must also fit.
        needed = max(
            math.floor(max(float(entry["required_width"]) for entry in members)) + 1,
            max(math.ceil(float(entry["scaled_width"])) for entry in members),
        )
        aligned = max(64, math.ceil(needed / align) * align)
        alternative[index] = min(widths[index], aligned)
    if sorted(set(alternative)) != alternative:
        raise ValueError("tighter widths are not strictly increasing")
    return alternative
