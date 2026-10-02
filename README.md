# Student Performance Prediction

End-to-end ML project: data cleaning -> encoding -> classification vs regression -> driver analysis -> deployed Streamlit app.

## Run
```bash
pip install -r requirements.txt
# put the real dataset in data/ (UCI "Student Performance": student-mat.csv or student-por.csv)
python train.py          # trains, compares models, saves models/ and reports/
streamlit run app.py
```
Without a CSV in `data/`, `train.py` falls back to synthetic data with the same schema (demo only).

## Design
- **Cleaning:** strips whitespace, removes duplicates, imputes inside the pipeline (median/mode, no leakage); before/after data-quality report.
- **Targets:** pass/fail (G3 >= 10) and final grade G3 (0-20).
- **Two scenarios:** `early` (no prior grades, usable early in the term, avoids leakage) and `with_grades` (adds G1, G2, the most accurate).
- **Models:** Logistic/Ridge, Random Forest, Gradient Boosting. 5-fold CV on train, final test on a held-out 20%. Best by CV score is refit on all data.
- **Drivers:** permutation importance on held-out data (`reports/feature_importance_*.png`).

## Results
Fill in from `reports/model_comparison.csv` after running on the real data. Report your own measured numbers.
