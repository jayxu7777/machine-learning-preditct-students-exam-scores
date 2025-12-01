#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 15:50:51 2025

@author: jiexu
"""

# ============================================================
# 90/10 holdout + (train-side) 10-fold × 2-repeats CV pipeline
# ============================================================

import pandas as pd
import numpy as np
from math import sqrt

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

# -----------------------------
# 0) Load your CSV here
# -----------------------------
df = pd.read_csv("Desktop/school_grades_dataset.csv") 
print("✅ Data loaded:", df.shape)
print(df.head())

# -----------------------------
# 1) Helpers
# -----------------------------
def make_labels(df, task):
    if task == "binary":
        return (df["G3"] >= 10).astype(int)
    elif task == "fivecls":
        bins = [-1, 9, 11, 13, 15, 20]
        return pd.cut(df["G3"], bins=bins, labels=[4,3,2,1,0]).astype(int)  # 0=I, 4=V
    else:
        return df["G3"].astype(float)

def task_metric(y_true, y_pred, task):
    return accuracy_score(y_true, y_pred) if task in ("binary", "fivecls") \
           else sqrt(mean_squared_error(y_true, y_pred))

class RFTopKSelector(BaseEstimator, TransformerMixin):
    """折内用 RF 重要度做 Top-K 特征选择（避免信息泄漏）"""
    def __init__(self, task="binary", top_k=20, random_state=0):
        self.task = task
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X, y):
        if self.task == "reg":
            rf = RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1)
        else:
            rf = RandomForestClassifier(n_estimators=300, random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        imp = rf.feature_importances_
        order = np.argsort(imp)[::-1]
        k = min(self.top_k, X.shape[1])
        self.keep_idx_ = order[:k]
        return self

    def transform(self, X):
        return X[:, self.keep_idx_]

def build_preprocessor(X, needs_std):
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler() if needs_std else "passthrough", num_cols),
        ],
        remainder="drop"
    )

def build_estimator(task, model_name):
    if model_name == "RF":
        return RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0) if task=="reg" \
               else RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
    if model_name == "XGB":
        return XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1) if task=="reg" \
               else XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                  eval_metric="logloss", n_jobs=-1)
    if model_name == "SVM":
        return SVR(kernel="rbf", C=2.0, gamma="scale") if task=="reg" \
               else SVC(kernel="rbf", C=2.0, gamma="scale", probability=False)
    if model_name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(128,), max_iter=1000, early_stopping=True, random_state=0) if task=="reg" \
               else MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, early_stopping=True, random_state=0)
    if model_name == "LR":   # Logistic Regression for classification only
        if task == "reg":
            raise ValueError("LR is logistic regression and only for classification tasks.")
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=1000, class_weight="balanced",
                                  multi_class="multinomial")
    if model_name == "GLM":  # Linear Regression for regression only
        if task != "reg":
            raise ValueError("GLM (LinearRegression) only for regression.")
        return LinearRegression(n_jobs=-1)
    raise ValueError(model_name)

def needs_standardization(model_name):
    return model_name in ("SVM", "MLP", "LR")  

# -------------------------------------------
# 2) Single 90/10 split (with stratify if cls)
# -------------------------------------------
TARGET = "G3"
X_all = df.drop(columns=[TARGET]).copy()
tasks = ["binary", "fivecls", "reg"]
models = ["RF", "XGB", "SVM", "MLP", "LR", "GLM"]

print("\n▶ Holdout = 90% train / 10% test ; Train side CV = 10-fold × 2 repeats\n")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.pipeline import Pipeline

for task in tasks:
    y_all = make_labels(df, task)

    # --- 90/10 split ---
    if task == "reg":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.10, random_state=42
        )
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.10, random_state=42, stratify=y_all
        )

    for model in models:
        if model == "LR" and task == "reg":
            continue
        if model == "GLM" and task != "reg":
            continue

        std_flag = needs_standardization(model)
        pre = build_preprocessor(X_tr, std_flag)
        selector = RFTopKSelector(
            task=("reg" if task == "reg" else "binary"),
            top_k=20,
            random_state=0
        )

        # --- estimator with no deprecated params ---
        if model == "LR":
            est = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced"
            )
        else:
            est = build_estimator(task, model)

        # ----------------------------
        # (A) Train-side CV (10×2)
        # ----------------------------
        if task == "reg":
            splitter = RepeatedKFold(n_splits=10, n_repeats=2, random_state=0).split(X_tr, y_tr)
        else:
            splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=0).split(X_tr, y_tr)

        cv_scores = []
        for tr_idx, va_idx in splitter:
            Xtr_f, Xva_f = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            ytr_f, yva_f = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

            pipe = Pipeline([
                ("pre", pre),
                ("sel", selector),
                ("est", est)
            ])
            pipe.fit(Xtr_f, ytr_f)
            yva_pred = pipe.predict(Xva_f)
            cv_scores.append(task_metric(yva_f, yva_pred, task))

        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))





        # ----------------------------
        # (B) Fit on full train (90%)
        # ----------------------------
        final_pipe = Pipeline([
            ("pre", pre),
            ("sel", selector),
            ("est", est)
        ])
        final_pipe.fit(X_tr, y_tr)

        # ----------------------------
        # (C) Evaluate once on 10% test
        # ----------------------------
        y_pred_test = final_pipe.predict(X_te)
        test_score = task_metric(y_te, y_pred_test, task)

        print(f"{task:8s} | {model:4s} | CV(10x2)={cv_mean:.3f}±{cv_std:.3f} | Test(10%)={test_score:.3f}")

# ================== EXTRAS: significance tests, NV baseline, metrics, visuals, A/B/C setups ==================
# Drop this whole block beneath your existing imports & helpers. It reuses your make_labels, task_metric,
# RFTopKSelector, build_preprocessor, build_estimator, needs_standardization.

import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# (0) Utilities: CI & t-tests
# -----------------------------
def mean_ci_95(values):
    arr = np.asarray(values, dtype=float)
    m = arr.mean()
    s = arr.std(ddof=1)
    n = len(arr)
    h = stats.t.ppf(0.975, df=n-1) * s / np.sqrt(n) if n > 1 and s > 0 else 0.0
    return m, h

def paired_t_marker(x, y, alpha=0.05):
    # two-sided paired t; return '⋆' if significant
    t, p = stats.ttest_rel(x, y)
    return "⋆" if (p is not None and p < alpha) else ""

def vs_baseline_marker(model_scores, base_scores, alpha=0.05):
    # dagger † if model significantly better than NV (paired across same folds)
    t, p = stats.ttest_rel(model_scores, base_scores)
    # for RMSE, "better" means LOWER; for PCC, "better" means HIGHER
    # We detect direction by comparing the means:
    if np.mean(model_scores) == np.mean(base_scores):
        return ""
    better = np.mean(model_scores) > np.mean(base_scores)  # assume PCC by default
    # we'll let caller pass already transformed scores (e.g., negate RMSE) if needed;
    # below we provide a safe wrapper.
    return "†" if (p is not None and p < alpha and better) else ""

# -----------------------------
# (1) Input setups A/B/C masks
# -----------------------------
def setup_columns(df, setup):
    cols = list(df.columns)
    if "G3" in cols:
        cols.remove("G3")
    if setup == "A":
        keep = cols
    elif setup == "B":
        keep = [c for c in cols if c != "G2"]
    elif setup == "C":
        keep = [c for c in cols if c not in ("G1", "G2")]
    else:
        raise ValueError(f"Unknown setup: {setup}")
    return keep

# -----------------------------
# (2) NV baseline per setup
# -----------------------------
def nv_predict_block(task, setup, X_fold, y_train_fold):
    """
    Returns NV predictions for a validation fold given setup A/B/C.
    - Classification:
        A: threshold/5-bin on G2
        B: threshold/5-bin on G1
        C: most frequent class (mode from TRAIN fold)
    - Regression:
        A: use G2
        B: use G1
        C: mean of TRAIN fold
    """
    if task == "reg":
        if setup == "A" and "G2" in X_fold.columns:
            return X_fold["G2"].to_numpy(dtype=float)
        elif setup == "B" and "G1" in X_fold.columns:
            return X_fold["G1"].to_numpy(dtype=float)
        else:
            return np.full(len(X_fold), float(np.mean(y_train_fold)))
    else:
        if setup == "A" and "G2" in X_fold.columns:
            z = X_fold["G2"].to_numpy()
            if task == "binary":
                return (z >= 10).astype(int)
            else:
                bins = [-1, 9, 11, 13, 15, 20]
                return np.digitize(z, bins) - 1  # 0..4
        elif setup == "B" and "G1" in X_fold.columns:
            z = X_fold["G1"].to_numpy()
            if task == "binary":
                return (z >= 10).astype(int)
            else:
                bins = [-1, 9, 11, 13, 15, 20]
                return np.digitize(z, bins) - 1
        else:
            # C: majority class from TRAIN fold
            mode = stats.mode(y_train_fold, keepdims=True)[0][0]
            return np.full(len(X_fold), mode)

# -----------------------------
# (3) Multi-metric wrappers
# -----------------------------
def compute_metrics(task, y_true, y_pred):
    if task in ("binary", "fivecls"):
        # PCC accuracy
        pcc = (y_true == y_pred).mean()
        return {"PCC": float(pcc)}
    else:
        rmse = sqrt(np.mean((y_true - y_pred) ** 2.0))
        return {"RMSE": float(rmse)}

# -----------------------------
# (4) Train-side CV with NV baseline & significance
#     - Uses 10x2 splits INSIDE the 90% train set
#     - Collects per-fold scores for each model and NV
#     - Adds markers: ⋆ (pairwise), † (vs NV)
# -----------------------------
def evaluate_with_significance(task, setup, X_train, y_train, model_name,
                               pre_builder, selector_cls, est_builder,
                               random_state=0):
    # Build single preprocessor object on TRAIN columns (fit inside folds by Pipeline)
    std_flag = needs_standardization(model_name)
    pre = pre_builder(X_train, std_flag)
    selector = selector_cls(task=("reg" if task == "reg" else "binary"), top_k=20, random_state=random_state)
    est = est_builder(task, model_name)

    # CV splitter
    if task == "reg":
        splitter = RepeatedKFold(n_splits=10, n_repeats=2, random_state=random_state).split(X_train, y_train)
        score_sign = -1.0  # for RMSE, lower is better -> we will flip sign for t-tests
    else:
        splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=random_state).split(X_train, y_train)
        score_sign = +1.0  # for PCC, higher is better

    model_scores = []
    base_scores  = []

    for tr_idx, va_idx in splitter:
        Xtr_f, Xva_f = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ytr_f, yva_f = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        # NV baseline for this fold
        yva_nv = nv_predict_block(task, setup, Xva_f, ytr_f)

        # ML pipeline
        pipe = Pipeline([("pre", pre), ("sel", selector), ("est", est)])
        pipe.fit(Xtr_f, ytr_f)
        yva_pred = pipe.predict(Xva_f)

        # fold scores
        m_model = compute_metrics(task, yva_f, yva_pred)
        m_base  = compute_metrics(task, yva_f, yva_nv)

        # we store the scalar for significance (flip sign for RMSE so "higher is better")
        if task in ("binary", "fivecls"):
            model_scores.append(m_model["PCC"])
            base_scores.append(m_base["PCC"])
        else:
            model_scores.append(-m_model["RMSE"])  # negate RMSE to align "bigger is better"
            base_scores.append(-m_base["RMSE"])

    # summarize (convert back to mean ± CI in original direction)
    model_arr = np.array(model_scores)
    base_arr  = np.array(base_scores)
    mean_m, h_m = mean_ci_95(model_arr)
    mean_b, h_b = mean_ci_95(base_arr)

    # back-transform means (RMSE was negated)
    if task in ("binary", "fivecls"):
        summary = {"mean": mean_m, "half_ci": h_m, "base_mean": mean_b, "base_half_ci": h_b}
    else:
        summary = {"mean": -mean_m, "half_ci": h_m, "base_mean": -mean_b, "base_half_ci": h_b}

    # significance markers
    dagger = vs_baseline_marker(model_arr, base_arr)  # † vs NV
    # (optional) pairwise marker you can compute against another model later with paired_t_marker

    return summary, model_arr, base_arr, dagger

# -----------------------------
# (5) Visuals on held-out test
#     - Confusion matrix for classification
#     - Scatter for regression
# -----------------------------
def plot_test_visuals(task, y_true, y_pred, title=""):
    if task in ("binary", "fivecls"):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format='d', cmap="Blues")
        plt.title(f"Confusion Matrix: {title}")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        plt.scatter(y_true, y_pred, s=18, alpha=0.7)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, linestyle="--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Regression scatter: {title}")
        plt.tight_layout()
        plt.show()

# -----------------------------
# (6) DRIVER EXAMPLE
#     - Loops tasks × setups(A/B/C) × models
#     - Inside train(90%): 10x2 CV for mean±CI + significance vs NV (†)
#     - On test(10%): plots visuals
# -----------------------------
def run_full_grid_with_significance(df, tasks, models, random_state=42):
    TARGET = "G3"
    results = []  # rows: task, setup, model, mean±CI, †

    for task in tasks:
        y_all = make_labels(df, task)
        for setup in ["A", "B", "C"]:
            used_cols = setup_columns(df, setup)
            X_all = df[used_cols].copy()

            # 90/10 holdout split (stratified for cls)
            if task == "reg":
                X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.10, random_state=random_state)
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.10,
                                                          random_state=random_state, stratify=y_all)

            # Evaluate NV on test (for reporting if you want)
            y_te_nv = nv_predict_block(task, setup, X_te, y_tr)
            nv_metric = compute_metrics(task, y_te, y_te_nv)
            if task in ("binary", "fivecls"):
                nv_disp = f"{nv_metric['PCC']:.3f}"
            else:
                nv_disp = f"{nv_metric['RMSE']:.3f}"

            # For each model
            for model in models:
                if model == "LR" and task == "reg":
                    continue
                if model == "GLM" and task != "reg":
                    continue

                # TRAIN-side CV mean±CI + † vs NV (CV folds)
                summary, model_cv_scores, base_cv_scores, dagger = evaluate_with_significance(
                    task, setup, X_tr, y_tr, model,
                    pre_builder=build_preprocessor,
                    selector_cls=RFTopKSelector,
                    est_builder=build_estimator,
                    random_state=0
                )

                # Refit on full train and test once for visuals
                std_flag = needs_standardization(model)
                pre = build_preprocessor(X_tr, std_flag)
                selector = RFTopKSelector(task=("reg" if task == "reg" else "binary"), top_k=20, random_state=0)
                est = build_estimator(task, model)
                pipe = Pipeline([("pre", pre), ("sel", selector), ("est", est)])
                pipe.fit(X_tr, y_tr)
                y_pred_te = pipe.predict(X_te)

                # Visuals (optional: comment out if running headless)
                plot_test_visuals(task, y_te, y_pred_te, title=f"{task} | {setup} | {model}")

                # format mean±CI string
                if task in ("binary", "fivecls"):
                    mean_disp = f"{summary['mean']:.3f} ± {summary['half_ci']:.3f}"
                else:
                    mean_disp = f"{summary['mean']:.3f} ± {summary['half_ci']:.3f}"

                results.append({
                    "Task": task,
                    "Setup": setup,
                    "Model": model,
                    "CV mean±CI": mean_disp,
                    "Sig vs NV (†)": dagger,
                    "NV(Test) metric": nv_disp
                })

    return pd.DataFrame(results)


# HOW TO CALL (example):
# ======================
# tasks = ["binary", "fivecls", "reg"]
# models = ["RF", "XGB", "SVM", "MLP", "LR", "GLM"]
# table = run_full_grid_with_significance(df, tasks, models, random_state=42)
# print(table)
# # You can also export:
# # table.to_csv("results_with_significance.csv", index=False)

table = run_full_grid_with_significance(df, tasks, models, random_state=42)
print(table)

table.to_csv("results_with_significance.csv", index=False)


# ==========================================================
# RF feature importance for ALL tasks (binary/fivecls/reg)
# across setups A/B/C; save CSVs and PNGs.
# ==========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def rf_importance_one(task, setup, df, out_dir="rf_importance_outputs", top_k=20, random_state=42):
    """
    For given task (binary/fivecls/reg) and setup (A/B/C):
    - 90/10 split on selected columns
    - fit RF on 90% train with one-hot for categoricals
    - compute relative importances (%), save CSV + PNG
    - return DataFrame with Feature, Importance, Task, Setup
    """
    os.makedirs(out_dir, exist_ok=True)

    # labels
    y_all = make_labels(df, task)

    # select columns per setup
    used_cols = setup_columns(df, setup)
    X_all = df[used_cols].copy()

    # split (stratified for classification)
    if task == "reg":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.10, random_state=random_state
        )
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.10, random_state=random_state, stratify=y_all
        )

    # numeric vs categorical
    num_cols = [c for c in X_tr.columns if X_tr[c].dtype != "object"]
    cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == "object"]

    # encoder only (no scaling; RF is scale-insensitive)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ], remainder="drop")

    X_enc = pre.fit_transform(X_tr)

    # RF model
    if task == "reg":
        rf = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)

    rf.fit(X_enc, y_tr)

    # names + importances
    feature_names = pre.get_feature_names_out()
    importances = rf.feature_importances_
    rel_imp = importances / importances.sum() * 100.0

    # top-k
    order = np.argsort(importances)[::-1][:min(top_k, len(importances))]
    df_top = pd.DataFrame({
        "Feature": feature_names[order],
        "Importance(%)": rel_imp[order]
    })
    df_top.insert(0, "Task", task)
    df_top.insert(1, "Setup", setup)

    # save CSV
    csv_path = os.path.join(out_dir, f"rf_importance_{task}_{setup}.csv")
    df_top.to_csv(csv_path, index=False)

    # plot PNG
    plt.figure(figsize=(9, 6))
    plt.barh(df_top["Feature"][::-1], df_top["Importance(%)"][::-1])
    plt.xlabel("Relative Importance (%)")
    plt.title(f"RF Feature Importance — Task: {task} | Setup: {setup}")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"rf_importance_{task}_{setup}.png")
    plt.savefig(png_path, dpi=160)
    plt.show()

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    return df_top

def rf_importance_all(df, out_dir="rf_importance_outputs", top_k=20, random_state=42):
    tasks = ["binary", "fivecls", "reg"]
    setups = ["A", "B", "C"]
    all_rows = []
    for task in tasks:
        for setup in setups:
            res = rf_importance_one(task, setup, df, out_dir=out_dir, top_k=top_k, random_state=random_state)
            all_rows.append(res)
    master = pd.concat(all_rows, ignore_index=True)
    master_path = os.path.join(out_dir, "rf_importance_MASTER.csv")
    master.to_csv(master_path, index=False)
    print(f"Master CSV saved: {master_path}")
    return master

# =========================
# RUN (examples):
# =========================
# master_table = rf_importance_all(df, out_dir="rf_importance_outputs", top_k=20, random_state=42)
# master_table.head()
master_table = rf_importance_all(df, out_dir="rf_importance_outputs", top_k=20, random_state=42)