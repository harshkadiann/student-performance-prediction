"""Synthetic stand-in for the UCI 'Student Performance' dataset (same schema).

Used only when no real CSV is found in data/. Replace with the real file
(student-mat.csv or student-por.csv) for your final results.
"""
import numpy as np
import pandas as pd


def make_synthetic(n=395, seed=42, messy=True):
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    studytime = np.clip(np.round(2 + 0.4 * ability + rng.normal(0, 0.8, n)), 1, 4).astype(int)
    failures = np.clip(np.round(np.maximum(0, -0.6 * ability + rng.normal(0, 0.6, n))), 0, 3).astype(int)
    absences = np.clip(rng.gamma(1.4, 4.0, n) - 0.8 * ability, 0, 75).round().astype(int)

    df = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], n, p=[.88, .12]),
        "sex": rng.choice(["F", "M"], n),
        "age": rng.integers(15, 22, n),
        "address": rng.choice(["U", "R"], n, p=[.78, .22]),
        "famsize": rng.choice(["GT3", "LE3"], n, p=[.71, .29]),
        "Pstatus": rng.choice(["T", "A"], n, p=[.9, .1]),
        "Medu": rng.integers(0, 5, n),
        "Fedu": rng.integers(0, 5, n),
        "Mjob": rng.choice(["at_home", "health", "other", "services", "teacher"], n),
        "Fjob": rng.choice(["at_home", "health", "other", "services", "teacher"], n),
        "reason": rng.choice(["course", "home", "other", "reputation"], n),
        "guardian": rng.choice(["mother", "father", "other"], n, p=[.7, .23, .07]),
        "traveltime": rng.integers(1, 5, n),
        "studytime": studytime,
        "failures": failures,
        "schoolsup": rng.choice(["yes", "no"], n, p=[.13, .87]),
        "famsup": rng.choice(["yes", "no"], n, p=[.61, .39]),
        "paid": rng.choice(["yes", "no"], n),
        "activities": rng.choice(["yes", "no"], n),
        "nursery": rng.choice(["yes", "no"], n, p=[.8, .2]),
        "higher": np.where(ability + rng.normal(0, 1, n) > -1.4, "yes", "no"),
        "internet": rng.choice(["yes", "no"], n, p=[.83, .17]),
        "romantic": rng.choice(["yes", "no"], n, p=[.33, .67]),
        "famrel": rng.integers(1, 6, n),
        "freetime": rng.integers(1, 6, n),
        "goout": rng.integers(1, 6, n),
        "Dalc": rng.integers(1, 6, n),
        "Walc": rng.integers(1, 6, n),
        "health": rng.integers(1, 6, n),
        "absences": absences,
    })
    base = (10.4 + 2.8 * ability + 0.5 * (studytime - 2) - 1.0 * failures
            - 0.06 * absences + 0.25 * (df["Medu"] - 2) + rng.normal(0, 1.2, n))
    g1 = np.clip(np.round(base + rng.normal(0, 0.8, n)), 0, 20)
    g2 = np.clip(np.round(0.6 * g1 + 0.4 * base + rng.normal(0, 0.9, n)), 0, 20)
    g3 = np.clip(np.round(0.8 * g2 + 0.2 * base + rng.normal(0, 0.9, n)), 0, 20)
    drop = rng.random(n) < 0.04          # a few "dropouts" scored 0, as in the real data
    g3 = np.where(drop, 0, g3)
    df["G1"], df["G2"], df["G3"] = g1.astype(int), g2.astype(int), g3.astype(int)

    if messy:
        for col in ["Medu", "Fedu", "studytime", "health", "Mjob", "Fjob", "internet", "absences"]:
            idx = rng.random(n) < 0.03
            df[col] = df[col].astype(object)
            df.loc[idx, col] = np.nan
        for col in ["Mjob", "Fjob", "reason"]:
            idx = rng.random(n) < 0.04
            df.loc[idx & df[col].notna(), col] = df.loc[idx & df[col].notna(), col] + " "
        df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)
    return df
