# Profiler-guided diagnostic enrichment pilot

This standalone Spotlights pilot starts with a real IOCR Pyinstrument profile,
inner profiler JSON, run log, subject source, and a frozen 44-page English/CPU
workload. It emits one direct `deepcopy` investigation and routes main recognition
through a targeted geometry probe. It does not edit IOCR production source or
claim an optimized implementation.

The recognition adapter keeps local copies of the routing and tighter-width
helpers from the earlier `profiler_recognition_bucket_probe/poc.py` and the
canonical digest helper from `targeted_font_size_probe/poc.py`. This preserves
the probe logic in a clean checkout.
Triage states the question before collection; it contains no expected crop or
padding count. Collection records only per-crop dimensions and effective width
configuration. Replay verifies that widths observed in IOCR's original
`predict_batch` calls match the adapter's routing calculation, then measures
crop distribution and padding. It cannot test model-output equality for a
different width policy from geometry alone.

## Run

Use the Python environment that imports IOCR's dependencies and set the local
weights path. The subject checkout must be clean at
`641b9806fcc38a33180c60396ac0b8df9a2b19e6`.

```sh
export IOCR_WEIGHTS_LOCATION=/path/to/IOCR_Weights/artifacts
export PYTHONDONTWRITEBYTECODE=1

python -m experiments.diagnostic_enrichment.pilot run \
  --iocr-repo /path/to/IOCR-spotlights \
  --images-dir /path/to/sdu_dba_45 \
  --profile /path/to/profile.txt \
  --profile-log /path/to/run.log \
  --inner-profile /path/to/profiling_raw.json \
  --profile-manifest /path/to/run_manifest.json \
  --run-dir experiments/diagnostic_enrichment/runs/new-pilot
```

The run performs baseline, selected capture, baseline in that order, then
validates and replays all capsules. `--run-dir` must be new. A failed step is
recorded in `status.json`; incomplete runs are kept for audit. To revalidate
an existing complete capture:

```sh
python -m experiments.diagnostic_enrichment.pilot replay \
  --run-dir experiments/diagnostic_enrichment/runs/new-pilot
```

`report.md` is the readable summary; `candidates.json`, `evaluation.json`,
`timing.json`, and `replay.json` are machine-readable. `intake.json`,
`suspects.json`, `reviewers/`, and `status.json` preserve the frozen input,
route decisions, evidence comparison, and wall time by step. Per-suspect
`direct/` and `probes/` sidecars preserve the plan and findings. The local
`capture/capture.json` manifest identifies each capsule but is not a published
candidate report. All files under `runs/` are Git-ignored.

The [first pilot result](RESULTS.md) records its Git-ignored local run and a
separate post-bracket Pyinstrument pass. That pass checks current hotspot costs;
the older pinned profile remains the prospective triage input.

## Contract and gates

- Schema version 1 and registered adapter name/version are required.
- Each capsule is at most 256 KiB; the complete collection is at most 2 MiB.
  Collection fails if either cap is exceeded, with no silent sampling.
- Capsules are marked `local_geometry_sensitive` with `local_only` retention.
  They contain crop geometry, not OCR text, image pixels, or tensors. Logs and
  complete page digests also remain local.
- Replay checks path confinement, exact listed files, byte count, SHA-256,
  schema, subject commit, workload hash, source hash, and effective config hash
  before diagnosing any invocation.
- The local semantic gate compares replayed bucket assignment with widths
  observed in the original IOCR call. The full-page gate compares complete raw
  OCR box digests across the adjacent control, capture, control sequence.
- Positive, negative, and inconclusive diagnostic outcomes are retained.
  A candidate remains an investigation until a concrete alternate policy
  passes model-output and page-level correctness gates. Any latency claim
  requires separate paired control, change, control whole-page runs.

Focused verification:

```sh
pytest -q tests/unit/test_diagnostic_enrichment_pilot.py
ruff check experiments/diagnostic_enrichment tests/unit/test_diagnostic_enrichment_pilot.py
```
