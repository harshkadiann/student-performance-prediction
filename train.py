"""End-to-end training: clean -> encode -> compare classifiers/regressors -> explain -> save."""
import argparse, glob, json, os, warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_gen import make_synthetic

warnings.filterwarnings("ignore")
SEED, PASS_MARK = 42, 10
GRADES = ["G1", "G2"]


def load(path):
    files = [path] if path else sorted(glob.glob("data/*.csv"))
    if files and os.path.exists(files[0]):
        print(f"Loading real data: {files[0]}")
        return pd.read_csv(files[0], sep=None, engine="python"), False
    print("!! No CSV in data/ -> using SYNTHETIC data (same schema). "
          "Metrics are NOT real results.")
    return make_synthetic(), True


def quality_report(df):
    """Share of cells affected by missing values, duplicate rows or stray whitespace."""
    cells = df.size
    missing = int(df.isna().sum().sum())
    dup_cells = int(df.duplicated().sum()) * df.shape[1]
    ws = int(sum((df[c].dropna().astype(str) != df[c].dropna().astype(str).str.strip()).sum()
                 for c in df.select_dtypes(exclude="number").columns))
    issues = missing + dup_cells + ws
    return {"missing_cells": missing, "duplicate_rows": int(df.duplicated().sum()),
            "whitespace_cells": ws, "issue_rate_pct": round(100 * issues / cells, 2)}


def clean(df):
    df = df.copy()
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype("string").str.strip().astype(object)
        df[c] = df[c].where(df[c].notna(), np.nan)
    for c in ["Medu", "Fedu", "studytime", "health", "absences"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates().reset_index(drop=True)


def preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])


CLF = {"Logistic Regression": lambda: LogisticRegression(max_iter=2000),
       "Random Forest": lambda: RandomForestClassifier(300, random_state=SEED),
       "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=SEED)}
REG = {"Ridge Regression": lambda: Ridge(alpha=1.0),
       "Random Forest": lambda: RandomForestRegressor(300, random_state=SEED),
       "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=SEED)}


def run_scenario(df, name, features, results, meta):
    X, g3 = df[features], df["G3"]
    y_clf = (g3 >= PASS_MARK).astype(int)
    Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = train_test_split(
        X, y_clf, g3, test_size=0.2, random_state=SEED, stratify=y_clf)

    best = {"clf": (None, -1, None), "reg": (None, -9, None)}
    for mname, mk in CLF.items():
        pipe = Pipeline([("prep", preprocessor(X)), ("m", mk())])
        cv = cross_val_score(pipe, Xtr, ytr_c, cv=5, scoring="accuracy").mean()
        pipe.fit(Xtr, ytr_c); p = pipe.predict(Xte)
        results.append({"scenario": name, "task": "classification", "model": mname,
                        "cv_score": round(cv, 4), "accuracy": round(accuracy_score(yte_c, p), 4),
                        "f1": round(f1_score(yte_c, p), 4)})
        if cv > best["clf"][1]:
            best["clf"] = (mname, cv, pipe)
    for mname, mk in REG.items():
        pipe = Pipeline([("prep", preprocessor(X)), ("m", mk())])
        cv = cross_val_score(pipe, Xtr, ytr_r, cv=5, scoring="r2").mean()
        pipe.fit(Xtr, ytr_r); p = pipe.predict(Xte)
        results.append({"scenario": name, "task": "regression", "model": mname,
                        "cv_score": round(cv, 4), "r2": round(r2_score(yte_r, p), 4),
                        "rmse": round(mean_squared_error(yte_r, p) ** 0.5, 3),
                        "mae": round(mean_absolute_error(yte_r, p), 3)})
        if cv > best["reg"][1]:
            best["reg"] = (mname, cv, pipe)

    # Feature drivers (permutation importance on held-out data, best classifier)
    cname, _, cpipe = best["clf"]
    pi = permutation_importance(cpipe, Xte, yte_c, n_repeats=15, random_state=SEED, scoring="accuracy")
    imp = pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)
    top = imp.head(10)[::-1]
    plt.figure(figsize=(7, 4.5)); plt.barh(top.index, top.values, color="#3b6fb6")
    plt.xlabel("Drop in accuracy when shuffled"); plt.title(f"Key drivers: {name} ({cname})")
    plt.tight_layout(); plt.savefig(f"reports/feature_importance_{name}.png", dpi=150); plt.close()

    # Refit winners on all data for deployment
    for task, key in [("clf", "classification"), ("reg", "regression")]:
        mname, _, pipe = best[task]
        pipe.fit(X, y_clf if task == "clf" else g3)
        joblib.dump(pipe, f"models/{task}_{name}.joblib")
        meta[f"{task}_{name}"] = mname
    meta["features"][name] = features
    meta["top_drivers"][name] = imp.head(6).round(4).to_dict()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None)
    raw, synthetic = load(ap.parse_args().data)
    before = quality_report(raw); df = clean(raw); after = quality_report(df)
    print("\nData quality before cleaning:", before)
    print("Data quality after cleaning (remaining NaNs are imputed inside the pipeline):", after)

    meta = {"synthetic": synthetic, "pass_mark": PASS_MARK, "quality_before": before,
            "quality_after": after, "features": {}, "top_drivers": {}}
    base = [c for c in df.columns if c not in GRADES + ["G3"]]
    num = df[base].select_dtypes(include="number")
    meta["defaults"] = {**num.median().to_dict(),
                        **{c: df[c].mode().iloc[0] for c in base if c not in num.columns}}
    meta["options"] = {c: sorted(df[c].dropna().unique().tolist()) for c in base if c not in num.columns}
    meta["ranges"] = {c: [int(df[c].min()), int(df[c].max())] for c in list(num.columns) + GRADES}

    results = []
    run_scenario(df, "early", base, results, meta)                 # no prior grades: usable early in term
    run_scenario(df, "with_grades", base + GRADES, results, meta)  # includes G1, G2
    res = pd.DataFrame(results); res.to_csv("reports/model_comparison.csv", index=False)
    json.dump(meta, open("models/meta.json", "w"), indent=2, default=str)
    print("\n", res.to_string(index=False))
    for k, v in meta["top_drivers"].items():
        print(f"\nTop drivers [{k}]:", v)


if __name__ == "__main__":
    main()
