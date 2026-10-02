"""Efficient hyper-parameter optimisation for the primary XGBoost model.

* Runs on the **TRAIN split only** (large datasets on a seeded stratified
  subsample, all fraud rows kept).
* Objective: mean **PR-AUC** (average precision) over stratified k-fold CV —
  the right metric for a rare positive class.
* Uses **Optuna** (TPE sampler + median pruner) when it is importable; otherwise
  falls back to scikit-learn's **HalvingRandomSearchCV** (successive halving).
* Search space is regularisation-forward (``min_child_weight``, ``reg_lambda``,
  ``reg_alpha``, ``gamma``, ``subsample``, ``colsample_bytree``) so the tuner
  cannot buy a fit by simply growing capacity.

Returns a dict with ``best_params`` plus a full trial record for
``tuning_<dataset>.json``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from config.seed import SEED

try:  # optional
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except Exception:  # pragma: no cover
    _HAS_OPTUNA = False


def _subsample(X, y, cap: int, seed: int):
    n = len(y)
    if cap <= 0 or n <= cap:
        return X, y, None
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    take_neg = max(1, cap - len(pos))
    keep = np.sort(np.concatenate([pos, rng.choice(neg, size=min(take_neg, len(neg)), replace=False)]))
    return X.iloc[keep], y[keep], int(len(keep))


def _fixed_kwargs(pos_weight: float) -> dict:
    return dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=pos_weight,
        random_state=SEED,
        n_jobs=0,
    )


# search space -------------------------------------------------------------- #
# Two regimes. `small` (few thousand rows, tens of fraud) forces shallow, heavily
# regularised trees so the tuner physically cannot memorise the positives — the
# train-vs-CV ROC-AUC gap stays inside the overfit threshold. `large` opens the
# space up because there are enough positives to support more capacity.
def _suggest_optuna(trial, small: bool) -> dict:
    if small:
        # Shallow-ish + a firm min_child_weight floor + real regularisation keeps
        # the train-vs-CV gap well under the 0.10 guard, while leaving the tuner
        # enough depth/rounds to stay discriminative on a hard, small dataset.
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
            "min_child_weight": trial.suggest_float("min_child_weight", 4.0, 25.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 8.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.05, 4.0, log=True),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 1200, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }


_HALVING_SPACE_LARGE = {
    "n_estimators": [150, 250, 400, 600, 800, 1000, 1200],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 4, 8, 12, 20],
    "reg_lambda": [0.1, 0.5, 1.0, 3.0, 8.0, 20.0],
    "reg_alpha": [0.0, 0.01, 0.1, 0.5, 2.0, 10.0],
    "gamma": [0.0, 0.05, 0.2, 0.5, 1.0, 3.0],
}
_HALVING_SPACE_SMALL = {
    "n_estimators": [100, 160, 250, 350, 450, 600],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.015, 0.02, 0.03, 0.05, 0.07],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [4, 6, 10, 15, 25],
    "reg_lambda": [0.5, 1.0, 3.0, 8.0, 20.0],
    "reg_alpha": [0.01, 0.1, 0.5, 2.0, 8.0],
    "gamma": [0.05, 0.2, 0.6, 1.5, 4.0],
}


def tune(dataset: str, X, y, *, n_trials: int, cv_folds: int, pos_weight: float,
         subsample_cap: int, small: bool | None = None) -> dict:
    y = np.asarray(y)
    if small is None:
        small = len(y) < 20_000 or int(y.sum()) < 400
    Xs, ys, sub_n = _subsample(X, y, subsample_cap, SEED)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    fixed = _fixed_kwargs(pos_weight)
    _HALVING_SPACE = _HALVING_SPACE_SMALL if small else _HALVING_SPACE_LARGE
    trials: list[dict] = []

    if _HAS_OPTUNA:
        def objective(trial):
            params = _suggest_optuna(trial, small)
            model = XGBClassifier(**params, **fixed)
            scores = []
            for step, (tr_i, va_i) in enumerate(skf.split(Xs, ys)):
                model.fit(Xs.iloc[tr_i], ys[tr_i])
                p = model.predict_proba(Xs.iloc[va_i])[:, 1]
                s = average_precision_score(ys[va_i], p)
                scores.append(s)
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        for t in study.trials:
            trials.append({"params": t.params, "score": (None if t.value is None else round(t.value, 5)),
                           "state": t.state.name})
        best_params = {**study.best_params}
        best_score = round(float(study.best_value), 5)
        method = f"optuna-tpe (n_trials={n_trials}, pruned={sum(1 for t in study.trials if t.state.name=='PRUNED')})"
    else:
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        search = HalvingRandomSearchCV(
            XGBClassifier(**fixed),
            _HALVING_SPACE,
            n_candidates=max(8, n_trials * 3),
            factor=3,
            resource="n_samples",
            scoring="average_precision",
            cv=skf,
            random_state=SEED,
            n_jobs=-1,
        )
        search.fit(Xs, ys)
        best_params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                       for k, v in search.best_params_.items()}
        best_score = round(float(search.best_score_), 5)
        method = f"halving-random-search (candidates={search.n_candidates_}, iters={search.n_iterations_})"
        for p, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
            if not np.isnan(s):
                trials.append({"params": p, "score": round(float(s), 5), "state": "COMPLETE"})

    return {
        "dataset": dataset,
        "method": method,
        "search_regime": "small (shallow, heavily regularised)" if small else "large",
        "optuna_available": _HAS_OPTUNA,
        "objective": "cv_mean_average_precision (PR-AUC)",
        "n_trials_requested": n_trials,
        "cv_folds": cv_folds,
        "tuned_on_rows": sub_n if sub_n is not None else int(len(ys)),
        "subsampled": sub_n is not None,
        "pos_weight": round(pos_weight, 4),
        "best_score": best_score,
        "best_params": best_params,
        "search_space": _HALVING_SPACE,
        "trials": trials[:200],
    }
