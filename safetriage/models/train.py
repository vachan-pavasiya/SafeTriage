"""Phase 2 — feature engineering, hyper-parameter tuning, fraud model, calibration.

Pipeline per dataset (IEEE-CIS first, then synthetic):

  1. fresh 60/20/20 stratified split (test never touched again here)
  2. feature engineering fit on TRAIN only  (safetriage/features.py)
  3. efficient hyper-parameter search on TRAIN via CV  (safetriage/tuning.py)
  4. PRIMARY XGBoost refit on TRAIN with the tuned params + early stopping on VAL
  5. SECONDARY Random Forest (fixed, regularised — the disagreement second opinion)
  6. probability calibration fit on a TRAIN sub-slice, reported on VAL
  7. overfit guard: (train ROC-AUC - CV ROC-AUC) gap
  8. FULL-refit artifacts (xgb_<ds>_full / rf_<ds>_full) on the WHOLE dataset with
     the tuned params — for serving only, never evaluated

Artifacts (safetriage/models/artifacts/, per dataset):
  xgb_<ds>.joblib  rf_<ds>.joblib  xgb_<ds>_full.joblib  rf_<ds>_full.joblib
  calibrator_<ds>.joblib  fe_<ds>.joblib  feature_spec_<ds>.json
  calibration_<ds>.json (+ .png)  model_metrics_<ds>.json  tuning_<ds>.json
  policy_thresholds_<ds>.json  uncertainty_norm_<ds>.json  split_summary_<ds>.json

Run:
  python -m safetriage.models.train                       # ieee then synthetic
  python -m safetriage.models.train ieee --epochs 20      # one dataset
  python -m safetriage.models.train --epochs 30 --cv-folds 4
"""

from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from config.seed import SEED, set_all_seeds
from safetriage import ARTIFACTS_DIR, DATA_DIR, load_policy_config
from safetriage.data.splits import make_splits, split_summary
from safetriage.features import fit_and_save_features
from safetriage.metrics import calibration_report, classification_metrics
from safetriage.tuning import tune

DATASETS = ("ieee", "synthetic")  # IEEE-CIS trained first


# --------------------------------------------------------------------------- #
# Base data (raw-ish frame + label; matrix is built by features.py AFTER split)
# --------------------------------------------------------------------------- #
def load_base(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    if dataset == "synthetic":
        df = pd.read_csv(DATA_DIR / "synthetic.csv")
    elif dataset == "ieee":
        df = pd.read_parquet(DATA_DIR / "processed_ieee.parquet")
    else:
        raise ValueError(dataset)
    return df, df["isFraud"].to_numpy()


def _rf(large: bool) -> RandomForestClassifier:
    # Fixed + bounded on purpose: RF is the second opinion for the disagreement
    # proxy, not the ranker, so it is not part of the hyper-parameter search.
    return RandomForestClassifier(
        n_estimators=200 if large else 400,
        max_depth=16 if large else None,
        min_samples_leaf=25 if large else 3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


def train_dataset(dataset: str, *, n_trials: int, cv_folds: int) -> dict:
    set_all_seeds(SEED)
    cfg = load_policy_config()
    tcfg = cfg["training"]
    df, y = load_base(dataset)
    large = len(df) > 100_000

    # 1. split — reuse the fresh split from scripts/build_splits.py if present,
    #    otherwise create it now (deterministic, seeded, identical either way).
    splits = make_splits(dataset, y, overwrite=False)
    tr, va = splits["train"], splits["val"]

    # 2. feature engineering — fit on TRAIN only ------------------------- #
    X = fit_and_save_features(dataset, df, tr)

    # calibration-fit holdout carved out of TRAIN
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr))
    cut = int(0.65 * len(tr))
    tr_fit, tr_cal = tr[perm[:cut]], tr[perm[cut:]]
    Xfit, yfit = X.iloc[tr_fit], y[tr_fit]
    Xcal, ycal = X.iloc[tr_cal], y[tr_cal]
    Xva, yva = X.iloc[va], y[va]

    imbalance = float((y[tr] == 0).sum()) / max(1.0, float((y[tr] == 1).sum()))
    pos_weight = float(np.sqrt(imbalance)) if large else 1.0

    # 3. hyper-parameter search on TRAIN -------------------------------------- #
    print(f"[{dataset}] tuning: {n_trials} trials x {cv_folds}-fold CV on the TRAIN split "
          f"({'subsampled ' if large else ''}pos_weight={pos_weight:.2f}) …")
    tuning = tune(dataset, X.iloc[tr], y[tr], n_trials=n_trials, cv_folds=cv_folds,
                  pos_weight=pos_weight, subsample_cap=int(tcfg["tune_subsample_cap"]))
    (ARTIFACTS_DIR / f"tuning_{dataset}.json").write_text(json.dumps(tuning, indent=2))
    bp = dict(tuning["best_params"])
    print(f"[{dataset}] best CV PR-AUC={tuning['best_score']}  params={bp}")

    fixed = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                 scale_pos_weight=pos_weight, random_state=SEED, n_jobs=0)

    # 4. PRIMARY XGBoost — refit on TRAIN ----------------------------------- #
    # Large set: a big ceiling + early stopping on VAL (thousands of fraud rows
    # there, so the stop signal is reliable).
    # Small set: the validation split has only ~20-30 fraud rows, so val AUC is
    # too noisy to early-stop on (it was stopping at ~16 rounds and under-fitting).
    # Instead trust the CV-selected `n_estimators` from the tuner directly, with a
    # generous early-stop only as a runaway guard.
    tuned_rounds = int(bp.pop("n_estimators", 600))
    if large:
        n_est = min(int(tcfg["boost_rounds_cap"]), max(tuned_rounds * 3, 400))
        esr = int(tcfg["early_stopping_rounds"])
    else:
        n_est = tuned_rounds
        esr = max(int(tcfg["early_stopping_rounds"]), tuned_rounds)  # ~never triggers
    xgb = XGBClassifier(**bp, **fixed, n_estimators=n_est, early_stopping_rounds=esr)
    xgb.fit(Xfit, yfit, eval_set=[(Xva, yva)], verbose=False)
    bi = getattr(xgb, "best_iteration", None)
    best_iteration = int(bi) if bi else (tuned_rounds if not large else n_est)
    if not large:  # for the small regime the CV-chosen budget is authoritative
        best_iteration = tuned_rounds

    # 5. SECONDARY Random Forest ------------------------------------------- #
    rf = _rf(large)
    rf.fit(Xfit, yfit)

    # 6. calibration — fit on the TRAIN calibration holdout, report on VAL --- #
    raw_va = xgb.predict_proba(Xva)[:, 1]
    raw_auc = roc_auc_score(yva, raw_va)
    cand = {}
    for method in ("sigmoid", "isotonic"):
        c = CalibratedClassifierCV(FrozenEstimator(xgb), method=method)
        c.fit(Xcal, ycal)
        cand[method] = {"cal": c,
                        "brier": brier_score_loss(ycal, c.predict_proba(Xcal)[:, 1]),
                        "auc": roc_auc_score(yva, c.predict_proba(Xva)[:, 1])}
    s, iso = cand["sigmoid"], cand["isotonic"]
    use_iso = large and (iso["brier"] <= 0.9 * s["brier"]) and (iso["auc"] >= raw_auc - 0.01)
    best_method = "isotonic" if use_iso else "sigmoid"
    best_cal = cand[best_method]["cal"]
    cal_va = best_cal.predict_proba(Xva)[:, 1]

    # 8. FULL-refit on the WHOLE dataset (serving only, never evaluated) ---- #
    xgb_full = XGBClassifier(**bp, **fixed, n_estimators=max(best_iteration, 50))
    xgb_full.fit(X, y, verbose=False)
    rf_full = _rf(large)
    rf_full.fit(X, y)

    # ---- persist model artifacts ---------------------------------------- #
    joblib.dump(xgb, ARTIFACTS_DIR / f"xgb_{dataset}.joblib")
    joblib.dump(rf, ARTIFACTS_DIR / f"rf_{dataset}.joblib")
    joblib.dump(best_cal, ARTIFACTS_DIR / f"calibrator_{dataset}.joblib")
    joblib.dump(xgb_full, ARTIFACTS_DIR / f"xgb_{dataset}_full.joblib")
    joblib.dump(rf_full, ARTIFACTS_DIR / f"rf_{dataset}_full.joblib")

    # ---- derived policy probability thresholds (val calibrated dist) ---- #
    prob_cfg = cfg["probability"]
    if prob_cfg.get("derive_from_data", False):
        p_low = float(np.percentile(cal_va, prob_cfg["p_low_percentile"]))
        p_high = float(np.percentile(cal_va, prob_cfg["p_high_percentile"]))
    else:
        p_low, p_high = float(prob_cfg["p_low"]), float(prob_cfg["p_high"])
    (ARTIFACTS_DIR / f"policy_thresholds_{dataset}.json").write_text(json.dumps({
        "dataset": dataset, "p_low": round(p_low, 6), "p_high": round(p_high, 6),
        "derivation": (f"validation calibrated-probability percentiles "
                       f"({prob_cfg['p_low_percentile']}, {prob_cfg['p_high_percentile']})"
                       if prob_cfg.get("derive_from_data") else "static config"),
    }, indent=2))

    # ---- disagreement-proxy normaliser (val) --------------------------- #
    rf_va = rf.predict_proba(Xva)[:, 1]
    disagree_va = np.abs(cal_va - rf_va)
    norm = {"p05": float(np.percentile(disagree_va, 5)), "p95": float(np.percentile(disagree_va, 95)),
            "max": float(disagree_va.max()), "mean": float(disagree_va.mean())}
    (ARTIFACTS_DIR / f"uncertainty_norm_{dataset}.json").write_text(json.dumps(norm, indent=2))

    # ---- calibration report (val) ------------------------------------- #
    cal_rep = calibration_report(yva, raw_va, cal_va, n_bins=10)
    cal_rep.update({"dataset": dataset, "method": best_method, "split": "validation", "seed": SEED})
    (ARTIFACTS_DIR / f"calibration_{dataset}.json").write_text(json.dumps(cal_rep, indent=2))
    _plot_calibration(cal_rep, ARTIFACTS_DIR / f"calibration_{dataset}.png")

    # 7. overfit guard + stable CV estimate over train+val -------------- #
    pool = np.sort(np.concatenate([tr, va]))
    cv_cap = 60_000
    if len(pool) > cv_cap:
        rs = np.random.default_rng(SEED)
        pos = pool[y[pool] == 1]
        neg = pool[y[pool] == 0]
        pool_cv = np.sort(np.concatenate([pos, rs.choice(neg, size=cv_cap - len(pos), replace=False)]))
        subsampled_from = int(len(pool))
    else:
        pool_cv, subsampled_from = pool, None
    Xpool, ypool = X.iloc[pool_cv], y[pool_cv]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    eval_xgb = XGBClassifier(**bp, **fixed, n_estimators=max(best_iteration, 50))
    cv_xgb = cross_val_predict(clone(eval_xgb), Xpool, ypool, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    cv_rf = cross_val_predict(clone(rf), Xpool, ypool, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    cv_xgb_auc = float(roc_auc_score(ypool, cv_xgb))
    train_auc = float(roc_auc_score(yfit, xgb.predict_proba(Xfit)[:, 1]))
    overfit_gap = round(train_auc - cv_xgb_auc, 5)
    overfit_warn = overfit_gap > float(tcfg["overfit_gap_warn"])

    cv_report = {
        "n": int(len(pool_cv)), "n_fraud": int(ypool.sum()), "folds": 5,
        "subsampled_from": subsampled_from,
        "xgb_roc_auc": round(cv_xgb_auc, 5),
        "xgb_pr_auc": round(float(average_precision_score(ypool, cv_xgb)), 5),
        "rf_roc_auc": round(float(roc_auc_score(ypool, cv_rf)), 5),
        "rf_pr_auc": round(float(average_precision_score(ypool, cv_rf)), 5),
    }
    overfit = {
        "train_roc_auc": round(train_auc, 5),
        "val_roc_auc": round(float(roc_auc_score(yva, cal_va)), 5),
        "cv_roc_auc": round(cv_xgb_auc, 5),
        "train_minus_cv_gap": overfit_gap,
        "gap_threshold": float(tcfg["overfit_gap_warn"]),
        "overfit_warning": bool(overfit_warn),
    }

    metrics = {
        "dataset": dataset, "seed": SEED,
        "n_features": int(X.shape[1]),
        "scale_pos_weight": round(pos_weight, 3),
        "tuned_params": bp,
        "boosting_rounds_used": best_iteration,
        "boosting_rounds_note": ("XGBoost has no 'epochs'; each boosting round adds one tree. "
                                 "The final fit uses early stopping on the validation split; "
                                 "the '--epochs' flag is the count of tuning configurations."),
        "hpo": {"method": tuning["method"], "trials": tuning["n_trials_requested"],
                "cv_folds": tuning["cv_folds"], "best_cv_pr_auc": tuning["best_score"],
                "tuned_on_rows": tuning["tuned_on_rows"]},
        "cv_train_val": cv_report,
        "overfit_check": overfit,
        "single_val_split_noisy": {
            "n_fraud_in_val": int(yva.sum()),
            "raw_xgb": classification_metrics(yva, raw_va),
            "calibrated_xgb": classification_metrics(yva, cal_va),
            "random_forest": classification_metrics(yva, rf_va),
        },
        "calibration_method": best_method,
        "disagreement_proxy_norm": norm,
    }
    (ARTIFACTS_DIR / f"model_metrics_{dataset}.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS_DIR / f"split_summary_{dataset}.json").write_text(json.dumps(split_summary(dataset, y), indent=2))

    flag = "  ⚠ OVERFIT WARNING" if overfit_warn else ""
    print(
        f"[{dataset}] calib={best_method}  Brier {cal_rep['brier_raw']:.4f}->{cal_rep['brier_calibrated']:.4f}  "
        f"ECE {cal_rep['ece_raw']:.4f}->{cal_rep['ece_calibrated']:.4f}  rounds={best_iteration}  "
        f"| CV ROC-AUC xgb={cv_report['xgb_roc_auc']:.3f} PR-AUC={cv_report['xgb_pr_auc']:.3f}  "
        f"| train-CV gap={overfit_gap:+.3f}{flag}"
    )
    return metrics


def _plot_calibration(rep: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    for key, label, colour in (
        ("reliability_raw", f"raw (Brier {rep['brier_raw']:.3f})", "#c1440e"),
        ("reliability_calibrated", f"calibrated (Brier {rep['brier_calibrated']:.3f})", "#1f6f4f"),
    ):
        pts = rep[key]
        ax.plot([p["mean_predicted"] for p in pts], [p["observed_frequency"] for p in pts],
                "o-", color=colour, label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud frequency")
    ax.set_title(f"Calibration — {rep['dataset']} ({rep['method']}, validation)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None, *, n_trials: int | None = None, cv_folds: int | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="ieee and/or synthetic (default: both, IEEE first)")
    ap.add_argument("--epochs", type=int, default=None, help="tuning configurations to evaluate (default: config hpo_trials)")
    ap.add_argument("--cv-folds", type=int, default=None, help="stratified CV folds per tuning trial")
    args = ap.parse_args(argv)

    tcfg = load_policy_config()["training"]
    trials = n_trials if n_trials is not None else (args.epochs if args.epochs is not None else int(tcfg["hpo_trials"]))
    folds = cv_folds if cv_folds is not None else (args.cv_folds if args.cv_folds is not None else int(tcfg["cv_folds"]))

    targets = args.datasets or list(DATASETS)
    for ds in targets:
        train_dataset(ds, n_trials=trials, cv_folds=folds)


if __name__ == "__main__":
    main()
