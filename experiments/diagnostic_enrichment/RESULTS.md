# First end-to-end pilot — October 11, 2026

**Result:** Spotlights routed two profiler suspects: one direct `deepcopy`
investigation and one core-recognition probe. The probe validated a
data-dependent padding condition on the pinned workload. It did not establish
an output-equivalent width change or a page-latency gain.

## Frozen scope and routing

- IOCR commit: `641b9806fcc38a33180c60396ac0b8df9a2b19e6`, clean checkout.
- Workload: 44 English/CPU JPEG pages, SHA-256
  `558395f6680224ddfc6ec6414a935af2bf2e9ae7380be7d819d29d91cf3a0a77`.
- Run ID: `20261011-first-pilot`. Local artifacts are under
  `experiments/diagnostic_enrichment/runs/20261011-first-pilot/` in the
  original worktree. The run is Git-ignored and is not included in this branch.
- The September 2 input profile took 177.059 s inside inference. It measured
  59.501 s inclusive in main recognition, including 44.893 s in ONNX, and
  3.919 s in the font-size `deepcopy` child. Those figures drove route selection.
  Its run log and inner profiler both cover all 44 pages; the inner profiler
  independently recorded 59.477 s in recognition.
- A fresh post-bracket profile took 174.716 s inside inference (176.905 s
  whole run). It measured 85.080 s in main recognition, 68.232 s in ONNX,
  and 3.976 s in the font-size `deepcopy`. The shift in sampled attribution
  reinforces the need to keep profile costs attached to their own run. IOCR's
  independent inner profiler recorded 85.069 s in recognition on that same
  fresh pass, closely matching the Pyinstrument parent cost.

The direct candidate points to input profile line 1077 and
`hrl_ocr/models/fonts_style_recognition/font_size_estimator.py:238`. It records
the measured copy and the unresolved ownership question. The probe candidate
points to input profile line 14 and the original bucket loop at
`hrl_ocr/models/recognition/ssocr/ssocr_core.py:177-185`.
The observed recognition configuration had height 32, padding enabled,
padding margin 1, and width buckets from 150 through 3000; the complete list
and its hash are in the local capture manifest.

## Fresh capture and diagnostic evidence

| Measure | Result |
| --- | ---: |
| Whole-run control, capture, control | 153.644, 169.171, 158.546 s |
| Capture versus control midpoint | +13.076 s (+8.38%) |
| Control drift | +4.901 s |
| Local geometry capsules | 44 calls, 68,070 bytes total |
| Latest manifest validation; diagnostic replay | 0.010 s; 0.012 s |
| Complete raw-page digest gate | 44/44 equal |
| Original routing assignment replay gate | 44/44 equal |
| Crops and assigned width | 6,467; 2,203,300 crop-width units |
| Padding from captured geometry and source resize rule | 513,765 units (23.32%) |
| Fallthrough; final configured bucket occupancy | 1 crop; 0 crops |
| Above maximum configured width | 0/6,467 crops (negative) |

The capsule contract is versioned and checks checksums, path confinement,
commit, workload, source, effective configuration, and byte caps before replay.
All local capsules were revalidated with the final contract. They contain crop
geometry only; the linked candidate and readable reports exclude OCR text,
pixels, tensors, and capsule contents.

The +13.076 s figure is a single-bracket estimate of capture overhead; the
controls themselves moved by 4.901 s. It is not an isolated per-hook cost.

The full raw-page digest gate protects the original instrumented run. The
local gate checks original bucket assignment against widths observed at IOCR's
`predict_batch` boundary. Padding comes from captured crop dimensions and
the source resize rule at `ssocr_core.py:131-140`. Geometry capsules cannot
test model-output equality for narrower buckets; that diagnostic is recorded
as **inconclusive**. No IOCR production source was edited.

## Evidence value and limits

The frozen profile-plus-source review could locate the copy, see that
recognition was costly, and identify fixed-width padding and the source-visible
last-bucket fallthrough. It could not establish this workload's crop
distribution or padding denominator. The probe added **one quantified runtime
condition** relative to that review and **one confirmation** of a source-visible
mechanism. The earlier standalone recognition-bucket PoC had already measured
this workload, so this pilot produced **zero new project findings**. The
profile-only snapshot was reconstructed after capture; no independent blinded
reviewer ran.

Routing counts: 2 inspected, 1 direct, 1 probed, 2 promoted as bounded
investigations, 0 rejected suspects, 0 inconclusive suspects. One subquestion
was negative (above-maximum crop count); one was inconclusive (alternate-width
output equality). Promotion does not mean that a replacement width policy is
correct or faster. The single-bracket timing difference estimates
instrumentation overhead. No latency speedup is claimed.

Pilot step wall times were 0.007 s intake, 162.430 s first control, 175.631 s
capture, 164.591 s second control, 0.037 s replay, and 0.003 s initial report
render; total 502.800 s. The separate fresh profile followed the bracket.
The replay figures in the table are from the final contract revalidation;
the 0.037 s step time belongs to the initial pilot run.

Audit files retained in the original worktree under the Git-ignored run
directory: `report.md`, `candidates.json`, `evaluation.json`, `timing.json`,
`replay.json`, `fresh_profile_summary.json`, `input_validation.json`, and
`status.json`. The local capture manifest and capsules remain there too.
