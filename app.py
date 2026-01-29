import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Loan Approval ML Dashboard", page_icon="🏦", layout="wide")

st.title("🏦 Loan Approval Prediction using Machine Learning")
st.markdown("### Logistic Regression with Probability Gauge")
st.markdown("---")

# -----------------------------
# LOAD & PREPROCESS DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")

    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna('Yes', inplace=True)
    df['Dependents'].fillna(0, inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(360, inplace=True)
    df['Credit_History'].fillna(1.0, inplace=True)

    df['Dependents'] = df['Dependents'].replace('[+]', '', regex=True).astype(int)

    return df

df = load_data()

# -----------------------------
# ENCODING
# -----------------------------
encoder = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Education', 'Married']]

y = df['Loan_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression()
model.fit(X_scaled, y)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📊 Navigation")
section = st.sidebar.radio("Go to", ["Dataset Overview", "Loan Approval Predictor"])

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
if section == "Dataset Overview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Model Accuracy")
    acc = model.score(X_scaled, y)
    st.success(f"Logistic Regression Accuracy: {round(acc * 100, 2)}%")

# -----------------------------
# GAUGE FUNCTION
# -----------------------------
def draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    ax.barh(0.5, 100, height=0.3, color="#E5E7E9")
    ax.barh(0.5, prob, height=0.3, color="#2ECC71" if prob >= 70 else "#F4D03F" if prob >= 40 else "#E74C3C")

    ax.text(prob, 0.5, f"{prob}%", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Loan Approval Probability")

    st.pyplot(fig)

# -----------------------------
# LOAN PREDICTOR
# -----------------------------
elif section == "Loan Approval Predictor":
    st.subheader("🧮 Check Your Loan Approval Probability")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("👤 Applicant Name")
        applicant_income = st.slider("💰 Applicant Income", 0, 30000, 5000)
        co_income = st.slider("🤝 Coapplicant Income", 0, 15000, 2000)
        loan_amt = st.slider("🏦 Loan Amount", 0, 600, 150)

    with col2:
        loan_term = st.selectbox("📆 Loan Term (Months)", [120, 180, 240, 300, 360])
        credit = st.radio("📊 Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x == 1.0 else "Bad")
        education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
        married = st.selectbox("💍 Married", ["Yes", "No"])

    if st.button("🔍 Predict Loan Approval"):
        input_data = pd.DataFrame([[
            applicant_income,
            co_income,
            loan_amt,
            loan_term,
            credit,
            1 if education == "Graduate" else 0,
            1 if married == "Yes" else 0
        ]], columns=X.columns)

        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        prob_percent = int(prob * 100)

        st.markdown(f"## 📊 Result for **{name}**")
        draw_gauge(prob_percent)

        if prob_percent >= 70:
            st.success("✅ High Chance of Loan Approval")
        elif prob_percent >= 40:
            st.warning("⚠️ Moderate Chance of Loan Approval")
        else:
            st.error("❌ Low Chance of Loan Approval")

        st.info("This prediction is generated using a trained Logistic Regression model.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 **ML-Powered Loan Approval Dashboard | Logistic Regression**")
