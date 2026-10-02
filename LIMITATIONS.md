# SafeTriage — Limitations

This is a **student research prototype**. Read every result with these caveats.

## Data

* **IEEE-CIS lacks the features SafeTriage is designed around.** No customer
  history, no "normal hour", no device/beneficiary provenance, no
  behavioural-consistency signal. On IEEE-CIS the interpretation layer runs in
  **reduced mode** using `C*`/`D*` columns as flagged proxies — this is a weaker
  version of the system, not the intended one.
* **The synthetic dataset is a controlled instrument, not evidence of production
  performance.** Its fraud typologies, prevalence, and the amount of class
  overlap are all authored. It is designed to make the false-positive/workload
  trade-off visible and measurable; it says nothing about how SafeTriage would
  behave on real traffic.
* **Small positive class.** The synthetic test split has ~23 fraud cases, so its
  recall/precision figures are noisy. Headline model discrimination is reported
  as cross-validated AUC to compensate, but the evaluation numbers still carry
  wide error bars.

## Method

* **The evaluation's analyst is perfect.** `STEP_UP_REVIEW` outcomes are resolved
  with the ground-truth label as a proxy — a real analyst queue has error,
  latency, and inconsistency. "Fraud caught in review" is an upper bound.
* **Audit outcomes are proxies.** Self-audit uses `isFraud` as a stand-in for a
  verified human re-review (`is_proxy=1`). Real audits would surface labelling
  errors this cannot.
* **Model disagreement is not epistemic uncertainty.** `|XGBoost − RandomForest|`
  says where two models diverge, not where the prediction is genuinely
  uncertain. Two models can agree and both be wrong.
* **Policy thresholds are derived, not optimised.** Probability cut-points come
  from validation-set percentiles; consistency/anomaly/uncertainty thresholds and
  all cost parameters are hand-set defaults, not tuned against an objective.
* **The baseline is deliberately naive.** A probability-only threshold picked for
  a fixed recall target is the textbook comparison, and on a weak model it blocks
  an unrealistic share of traffic — which is the motivation for SafeTriage, but
  it also flatters the comparison. A stronger baseline (cost-sensitive threshold,
  a second model) would narrow the gap.

## Scope

* **Graph analysis is illustrative only** — a small structure drawn from two
  synthetic columns. No graph database, no GraphSAGE, no message passing.
* **Next-best-evidence is template-based** — fixed rules over the transaction's
  own fields, no ranking against historical analyst resolutions, no LLM.
* **Feedback is stored, not learned from.** The recalibration loop is architected
  (schema + capture) but not run; the MVP does not retrain on analyst verdicts.
* **Transaction "replay" is static.** Records are scored from a file, not
  consumed from a stream.

## Production gap

A real deployment would additionally need: privacy controls and data
minimisation, authn/authz and audit logging, drift and performance monitoring,
verified label pipelines, human-factors testing of the review UI, latency SLAs
under streaming load, and real infrastructure (feature store, queue, graph store)
in place of the local files and SQLite used here.
