import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Approval App", layout="wide")

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visual Analysis", "Loan Prediction"])

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")

    # Handling missing values
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna('Yes')
    df['Dependents'] = df['Dependents'].fillna(0)
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    df['LoanAmount'] = df['LoanAmount'].fillna(128.0)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360.0)
    df['Credit_History'] = df['Credit_History'].fillna(1.0)

    df['Dependents'] = df['Dependents'].replace('[+]', '', regex=True).astype(int)
    return df

df = load_data()

# ---------------- Data Exploration ----------------
if page == "Data Exploration":
    st.title("📊 Loan Dataset Overview")
    st.subheader("Raw Dataset")
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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Status vs Applicant Income")
        fig, ax = plt.subplots()
        sb.boxplot(x=df['Loan_Status'], y=df['ApplicantIncome'], ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Loan Status vs Coapplicant Income")
        fig, ax = plt.subplots()
        sb.barplot(x=df['Loan_Status'], y=df['CoapplicantIncome'], ax=ax)
        st.pyplot(fig)

    st.subheader("Loan Status vs Credit History")
    fig, ax = plt.subplots()
    pd.crosstab(df['Loan_Status'], df['Credit_History']).plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Property Area vs Loan Status")
    fig, ax = plt.subplots()
    pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
    st.pyplot(fig)

# ---------------- Loan Prediction ----------------
elif page == "Loan Prediction":
    st.title("🏦 Loan Approval Prediction")

    # Encode categorical data
    df_model = df.copy()
    le = LabelEncoder()
    for col in ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']:
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop('Loan_Status', axis=1)
    y = df_model['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model Accuracy: {acc*100:.2f}%")

    st.subheader("Enter Applicant Details")

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
        loan_term = st.number_input("Loan Amount Term", 12, 480, 360)

    credit_hist = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

    if st.button("Predict Loan Status"):
        input_data = pd.DataFrame([[gender, married, dependents, education, self_emp,
                                    applicant_income, co_income, loan_amt,
                                    loan_term, credit_hist, property_area]],
                                  columns=X.columns)

        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                input_data[col] = le.fit_transform(input_data[col])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")
