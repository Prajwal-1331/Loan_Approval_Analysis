import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["Data Exploration", "Visual Analysis", "Loan Prediction"])

# ---------------- Load & Preprocess Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")

    # Drop Loan_ID (not useful for ML)
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # Fill missing values
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna('Yes')
    df['Dependents'] = df['Dependents'].fillna(0)
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    df['LoanAmount'] = df['LoanAmount'].fillna(128.0)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360.0)
    df['Credit_History'] = df['Credit_History'].fillna(1.0)

    # Clean Dependents column
    df['Dependents'] = df['Dependents'].replace('[+]', '', regex=True).astype(int)

    return df

df = load_data()

# ---------------- Encode Data for Model ----------------
def encode_data(df):
    df_model = df.copy()

    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Loan_Status': {'Y': 1, 'N': 0}
    }

    for col, mapping in mappings.items():
        df_model[col] = df_model[col].map(mapping)

    return df_model

# ---------------- Train Model ----------------
@st.cache_resource
def train_model(df):
    df_model = encode_data(df)

    X = df_model.drop('Loan_Status', axis=1)
    y = df_model['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model(df)

# ---------------- Data Exploration ----------------
if page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.dataframe(df)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Statistical Summary")
    st.write(df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']].describe())

# ---------------- Visual Analysis ----------------
elif page == "Visual Analysis":
    st.title("📈 Visual Analysis")

    fig, ax = plt.subplots()
    sb.boxplot(x=df['Loan_Status'], y=df['ApplicantIncome'], ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sb.barplot(x=df['Loan_Status'], y=df['CoapplicantIncome'], ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    pd.crosstab(df['Loan_Status'], df['Credit_History']).plot(kind='bar', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
    st.pyplot(fig)

# ---------------- Loan Prediction ----------------
elif page == "Loan Prediction":
    st.title("🏦 Loan Approval Prediction")
    st.success(f"Model Accuracy: {accuracy*100:.2f}%")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        married = st.selectbox("Married", ['Yes', 'No'])
        dependents = st.number_input("Dependents", 0, 5, 0)

    with col2:
        education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
        self_emp = st.selectbox("Self Employed", ['Yes', 'No'])
        applicant_income = st.number_input("Applicant Income", 0)

    with col3:
        co_income = st.number_input("Coapplicant Income", 0)
        loan_amt = st.number_input("Loan Amount", 0)
        loan_term = st.number_input("Loan Term", 12, 480, 360)

    credit_hist = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

    if st.button("Predict Loan Status"):
        input_data = pd.DataFrame([[
            1 if gender == 'Male' else 0,
            1 if married == 'Yes' else 0,
            dependents,
            1 if education == 'Graduate' else 0,
            1 if self_emp == 'Yes' else 0,
            applicant_income,
            co_income,
            loan_amt,
            loan_term,
            credit_hist,
            2 if property_area == 'Urban' else 1 if property_area == 'Semiurban' else 0
        ]], columns=model.feature_names_in_)

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")
