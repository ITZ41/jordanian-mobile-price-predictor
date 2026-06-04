"""Model training, selection, tuning, and evaluation.

Trains 3 gradient boosting models (HistGBM, XGBoost, LightGBM),
selects the winner via 5-fold CV, fine-tunes with grid search,
and generates SHAP feature importance + learning curves.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
import json
import joblib

from src.features import Preprocessor
from src.paths import CLEANED_DATA_PATH, MODEL_PATH, METRICS_PATH, REPORTS_DIR
from src.config import get as cfg
from src.log import get as get_log

log = get_log("train_model")


def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare training data, running validation first."""
    from src.validate import validate_and_report

    log.info("Running data validation...")
    valid = validate_and_report(CLEANED_DATA_PATH)
    if not valid:
        log.error("Data validation failed — aborting training")
        sys.exit(1)

    df = pd.read_csv(CLEANED_DATA_PATH)
    df["storage_gb"] = df["storage_gb"].fillna(df["storage_gb"].median())
    df["series"] = df["series"].fillna("Other")
    df["model_name"] = df["model_name"].fillna("Unknown")
    df["color_actual"] = df["color_actual"].fillna("Unknown")
    if "phone_age_months" in df.columns:
        df["phone_age_months"] = df["phone_age_months"].fillna(
            df["phone_age_months"].median()
        )
    else:
        df["phone_age_months"] = 0

    feature_cols = [
        c
        for c in [
            "title", "brand", "series", "storage_gb",
            "phone_age_months", "condition", "model_name", "color_actual",
        ]
        if c in df.columns
    ]
    X = df[feature_cols]
    y = df["price_jd"]
    return X, y


def _r2_orig(estimator, X, y_log):
    y_true = np.expm1(y_log)
    y_pred = np.expm1(estimator.predict(X))
    return r2_score(y_true, y_pred)


def _mae_orig(estimator, X, y_log):
    y_true = np.expm1(y_log)
    y_pred = np.expm1(estimator.predict(X))
    return mean_absolute_error(y_true, y_pred)


def _get_candidates() -> dict[str, Pipeline]:
    """Build the 3 candidate pipelines."""
    hist_params = {
        "random_state": 42,
        "max_iter": cfg("model.hist_gradient_boosting.max_iter", 400),
        "learning_rate": cfg("model.hist_gradient_boosting.learning_rate", 0.03),
        "max_depth": cfg("model.hist_gradient_boosting.max_depth", 8),
        "l2_regularization": cfg("model.hist_gradient_boosting.l2_regularization", 1.0),
    }
    xgb_params = {
        "random_state": 42,
        "n_estimators": cfg("model.xgboost.n_estimators", 500),
        "learning_rate": cfg("model.xgboost.learning_rate", 0.05),
        "max_depth": cfg("model.xgboost.max_depth", 6),
        "subsample": cfg("model.xgboost.subsample", 0.8),
        "objective": "reg:squarederror",
        "verbosity": 0,
    }
    lgb_params = {
        "random_state": 42,
        "n_estimators": cfg("model.lightgbm.n_estimators", 500),
        "learning_rate": cfg("model.lightgbm.learning_rate", 0.05),
        "max_depth": cfg("model.lightgbm.max_depth", 6),
        "subsample": cfg("model.lightgbm.subsample", 0.8),
        "verbose": -1,
    }

    candidates: dict[str, Pipeline] = {
        "HistGradientBoosting": Pipeline([
            ("prep", Preprocessor()),
            ("reg", HistGradientBoostingRegressor(**hist_params)),
        ]),
        "XGBoost": Pipeline([
            ("prep", Preprocessor()),
            ("reg", XGBRegressor(**xgb_params)),
        ]),
    }

    # LightGBM is optional — only add if installed
    try:
        from lightgbm import LGBMRegressor
        candidates["LightGBM"] = Pipeline([
            ("prep", Preprocessor()),
            ("reg", LGBMRegressor(**lgb_params)),
        ])
        log.info("LightGBM available — added to candidates")
    except ImportError:
        log.warning("LightGBM not installed — skipping. pip install lightgbm")

    return candidates


def _log_experiment(name: str, params: dict, mae: float, r2: float, stage: str) -> None:
    """Append experiment result to experiment_log.jsonl."""
    log_path = MODEL_PATH.parent / "experiment_log.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": name,
        "stage": stage,
        "params": params,
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"Experiment logged: {name} ({stage}) MAE={mae:.2f} R²={r2:.3f}")


def _plot_learning_curve(best_pipe: Pipeline, X: pd.DataFrame, y_log: np.ndarray) -> None:
    """Generate and save learning curve plot."""
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            best_pipe, X, y_log,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
        )
        train_mae = -train_scores.mean(axis=1)
        val_mae = -val_scores.mean(axis=1)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_mae, "o-", color="#58A6FF", label="Training MAE")
        ax.fill_between(train_sizes, train_mae - train_scores.std(axis=1),
                         train_mae + train_scores.std(axis=1), alpha=0.1, color="#58A6FF")
        ax.plot(train_sizes, val_mae, "o-", color="#F85149", label="Validation MAE")
        ax.fill_between(train_sizes, val_mae - val_scores.std(axis=1),
                         val_mae + val_scores.std(axis=1), alpha=0.1, color="#F85149")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("MAE (JOD)")
        ax.set_title("Learning Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = REPORTS_DIR / "learning_curve.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Learning curve saved to {fig_path}")
    except Exception as e:
        log.warning(f"Could not generate learning curve: {e}")


def _shap_importance(best_pipe: Pipeline, X: pd.DataFrame, y_log: np.ndarray) -> None:
    """Generate SHAP feature importance plot and summary JSON."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Get the fitted preprocessor and regressor from the pipeline
        preprocessor = best_pipe.named_steps["prep"]
        regressor = best_pipe.named_steps["reg"]

        # Transform data using the preprocessor
        X_transformed = preprocessor.transform(X)

        # Get feature names
        title_feat_names = [
            "is_pro", "is_max", "is_ultra", "is_plus",
            "has_warranty", "is_sealed", "is_new", "is_5g",
            "is_esim", "ram_gb", "battery_pct",
        ]
        cat_feat_names = ["brand", "series", "condition", "model_name", "color_actual"]
        num_feat_names = [c for c in ["storage_gb", "phone_age_months"] if c in X.columns]
        feature_names = title_feat_names + cat_feat_names + num_feat_names

        # Trim to actual number of columns
        n_cols = X_transformed.shape[1]
        feature_names = feature_names[:n_cols]

        # Compute SHAP values
        if hasattr(regressor, "predict"):
            explainer = shap.Explainer(regressor.predict, X_transformed[:50])
            shap_values = explainer(X_transformed[:200])

            # Beeswarm plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.beeswarm(shap_values, show=False, max_display=15)
            fig = plt.gcf()
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            shap_path = REPORTS_DIR / "shap_importance.png"
            fig.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"SHAP plot saved to {shap_path}")

            # Summary JSON
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            importance = {
                feature_names[i]: round(float(mean_abs[i]), 4)
                for i in range(len(feature_names))
            }
            importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
            imp_path = MODEL_PATH.parent / "feature_importance.json"
            with open(imp_path, "w") as f:
                json.dump(importance, f, indent=2)
            log.info(f"Feature importance saved to {imp_path}")
        else:
            log.warning("Regressor has no predict method — skipping SHAP")

    except ImportError:
        log.warning("shap not installed — skipping SHAP. pip install shap")
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")


def train() -> None:
    """Main training pipeline."""
    log.info("Loading data...")
    X, y = _load_data()
    y_log = np.log1p(y)
    log.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

    # --- Define candidates ---
    candidates = _get_candidates()

    # --- Evaluate with 5-fold CV ---
    cv = KFold(
        n_splits=cfg("model.tuning.cv_folds", 5),
        shuffle=True, random_state=42,
    )
    best_name = None
    best_cv_mae = float("inf")
    best_pipeline = None

    for name, pipe in candidates.items():
        log.info(f"Evaluating {name} with {cv.get_n_splits()}-fold CV...")
        scores = cross_validate(
            pipe, X, y_log, cv=cv,
            scoring={"r2": r2_orig, "mae": _mae_orig},
            n_jobs=1,
        )
        cv_mae = scores["test_mae"].mean()
        cv_r2 = scores["test_r2"].mean()
        log.info(f"  CV R²: {cv_r2:.3f} (+/- {scores['test_r2'].std():.3f})")
        log.info(f"  CV MAE: {cv_mae:.2f} (+/- {scores['test_mae'].std():.2f}) JOD")

        _log_experiment(name, {}, cv_mae, cv_r2, stage="cv_comparison")

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_name = name
            best_pipeline = pipe

    log.info(f"Best model: {best_name} (CV MAE: {best_cv_mae:.2f} JOD)")

    # --- Tune the winner ---
    log.info(f"Tuning {best_name} hyperparameters...")
    param_grid = cfg("model.tuning.param_grid", {
        "reg__max_iter": [400, 600, 800],
        "reg__learning_rate": [0.02, 0.03],
        "reg__max_depth": [7, 8, 10],
        "reg__l2_regularization": [0.1, 1.0, 5.0],
    })

    grid = GridSearchCV(
        best_pipeline, param_grid, cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=1,
    )
    grid.fit(X, y_log)

    log.info(f"Best params: {grid.best_params_}")
    log.info(f"Best CV score: {-grid.best_score_:.2f} JOD")
    best = grid.best_estimator_

    _log_experiment(best_name, grid.best_params_, -grid.best_score_,
                     0, stage="tuned")

    # --- Final test evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    best.fit(X_train, np.log1p(y_train))
    y_pred = np.expm1(best.predict(X_test))

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae_std = np.std(np.abs(y_test - y_pred))

    log.info(f"Test Set Performance: MAE={mae:.2f} +/- {mae_std:.2f} JOD, R²={r2:.3f}")

    _log_experiment(best_name, grid.best_params_, mae, r2, stage="final_test")

    # --- Save model ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    log.info(f"Model saved to {MODEL_PATH}")

    # --- Save metrics ---
    metrics = {
        "model": best_name,
        "mae": round(mae, 2),
        "mae_std": round(mae_std, 2),
        "r2": round(r2, 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "training_date": datetime.now().isoformat(),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {METRICS_PATH}")

    # --- Generate learning curves ---
    log.info("Generating learning curve...")
    _plot_learning_curve(best, X, y_log)

    # --- Generate SHAP feature importance ---
    log.info("Computing SHAP feature importance...")
    _shap_importance(best, X, y_log)

    log.info("Training complete!")


if __name__ == "__main__":
    train()
