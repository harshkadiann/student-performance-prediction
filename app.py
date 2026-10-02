"""Streamlit app: predict pass/fail and final grade, and show the main drivers.
Run:  streamlit run app.py   (after `python train.py`)
"""
import json
import joblib
import pandas as pd
import streamlit as st

META = json.load(open("models/meta.json"))
CORE = ["failures", "studytime", "absences", "higher", "Medu", "Fedu", "goout",
        "Walc", "age", "sex", "internet", "schoolsup", "famsup", "paid", "health"]


@st.cache_resource
def load_models(scenario):
    return joblib.load(f"models/clf_{scenario}.joblib"), joblib.load(f"models/reg_{scenario}.joblib")


def predict(row, scenario):
    clf, reg = load_models(scenario)
    X = pd.DataFrame([row])[META["features"][scenario]]
    return float(clf.predict_proba(X)[0, 1]), float(reg.predict(X)[0])


st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
if META.get("synthetic"):
    st.warning("Models were trained on synthetic demo data. Retrain on the real dataset for real results.")

use_grades = st.toggle("I know the student's first two period grades (G1, G2)")
scenario = "with_grades" if use_grades else "early"
row = dict(META["defaults"])

cols = st.columns(2)
for i, f in enumerate(CORE):
    with cols[i % 2]:
        if f in META["options"]:
            opts = META["options"][f]
            row[f] = st.selectbox(f, opts, index=opts.index(META["defaults"][f]))
        else:
            lo, hi = META["ranges"][f]
            row[f] = st.slider(f, lo, hi, int(META["defaults"][f]))
if use_grades:
    for g in ["G1", "G2"]:
        lo, hi = META["ranges"][g]
        row[g] = st.slider(f"{g} (0-20)", lo, hi, 10)

if st.button("Predict", type="primary"):
    p_pass, grade = predict(row, scenario)
    st.metric("Probability of passing (final grade >= %d)" % META["pass_mark"], f"{p_pass:.0%}")
    st.metric("Predicted final grade (0-20)", f"{grade:.1f}")
    if p_pass < 0.5:
        st.error("At risk: consider tutoring, attendance follow-up and a mentor check-in.")
    else:
        st.success("On track. Keep monitoring attendance and study habits.")
    st.subheader("Main drivers in this model")
    st.bar_chart(pd.Series(META["top_drivers"][scenario]))
