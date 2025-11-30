import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Stroke Risk Dashboard", layout="wide")

def find_data_path():
    candidates = [
        "data/healthcare-dataset-stroke-data.csv",
        "healthcare-dataset-stroke-data.csv",
        "./data/healthcare-dataset-stroke-data.csv",
        "./healthcare-dataset-stroke-data.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

data_path = find_data_path()
if data_path is None:
    st.error("Please upload healthcare-dataset-stroke-data.csv")
    st.stop()

df = pd.read_csv(data_path)

model_path = "stroke_rf_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.info("No pretrained model found. Training Random Forest...")
    X = pd.get_dummies(df.drop(columns=["stroke"], errors='ignore'))
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    st.success(f"Trained model — Accuracy: {acc:.2f}, AUC: {auc:.2f}")

st.title("Stroke Risk Prediction Dashboard")
st.sidebar.header("Patient Info")
age = st.sidebar.slider("Age", int(df['age'].min()), int(df['age'].max()), 50)
glucose = st.sidebar.number_input("Average Glucose Level", float(df['avg_glucose_level'].min()), float(df['avg_glucose_level'].max()), 100.0)
bmi = st.sidebar.number_input("BMI", float(df['bmi'].min()), float(df['bmi'].max()), 25.0)
hypertension = st.sidebar.selectbox("Hypertension", [0,1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0,1])
gender = st.sidebar.selectbox("Gender", df['gender'].unique())
smoking_status = st.sidebar.selectbox("Smoking Status", df['smoking_status'].unique())

input_df = pd.DataFrame({'age':[age],'avg_glucose_level':[glucose],'bmi':[bmi],
                         'hypertension':[hypertension],'heart_disease':[heart_disease],
                         'gender':[gender],'smoking_status':[smoking_status]})
input_df = pd.get_dummies(input_df)
missing_cols = set(model.feature_names_in_) - set(input_df.columns)
for c in missing_cols:
    input_df[c] = 0
input_df = input_df[model.feature_names_in_]

risk_prob = model.predict_proba(input_df)[:,1][0]
st.subheader(f"Predicted Stroke Risk: {risk_prob:.2%}")
if risk_prob < 0.2:
    st.success("Low Risk")
elif risk_prob < 0.5:
    st.warning("Moderate Risk")
else:
    st.error("High Risk")

# Example population chart
df['age_group'] = pd.cut(df['age'], bins=[0,20,40,60,80,120], labels=["0-20","21-40","41-60","61-80","80+"] )
stroke_by_age = df.groupby('age_group')['stroke'].mean().reset_index()
fig_age = px.bar(stroke_by_age, x='age_group', y='stroke', title="Stroke Probability by Age Group", labels={'stroke':'Stroke Rate'})
st.plotly_chart(fig_age, use_container_width=True)
