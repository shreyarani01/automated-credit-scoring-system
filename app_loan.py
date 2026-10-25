import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Wise - Loan Approval Predictor", 
    page_icon="💳", 
    layout="centered"
)

# Application Header
st.title("💳 Credit Wise: Loan Approval Predictor")
st.caption("Evaluate applicant financial profiles in real-time using a benchmarked Gaussian Naive Bayes model.")
st.write("---")

@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_ml_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

nb_model, scaler = load_artifacts()

# User Input Section
st.subheader("📋 Applicant Information")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score", 
            min_value=300, 
            max_value=850, 
            value=700,
            help="FICO credit score range (300 - 850)"
        )
        applicant_income_annual = st.number_input(
            "Applicant Annual Income ($)", 
            min_value=0, 
            value=75000,
            step=1000
        )

    with col2:
        dti_percent = st.number_input(
            "Debt-to-Income (DTI) Ratio (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=15.0, 
            step=0.1,
            help="Total monthly debt payments divided by gross monthly income"
        )

# Preprocessing to align with notebook training scale
applicant_income = applicant_income_annual / 12.0
dti_ratio = dti_percent / 100.0
dti_sq = dti_ratio ** 2
credit_score_sq = credit_score ** 2

st.write("")
predict_btn = st.button("⚡ Predict Loan Status", type="primary", use_container_width=True)

# Output Section
if predict_btn:
    # Construct input DataFrame matching exact feature schema
    input_df = pd.DataFrame([{
        'Applicant_Income': applicant_income,
        'DTI_Ratio': dti_ratio,
        'Credit_Score': credit_score,
        'DTI_Ratio_sq': dti_sq,
        'Credit_Score_sq': credit_score_sq
    }])
    
    # Live inference
    scaled_input = scaler.transform(input_df)
    prediction = nb_model.predict(scaled_input)[0]
    prob = nb_model.predict_proba(scaled_input)[0][1]

    st.write("---")
    st.subheader("📊 Prediction Assessment")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if prediction == 1:
            st.success("### ✅ Loan Approved")
            st.caption("Applicant meets the credit threshold for approval.")
        else:
            st.error("### ❌ Loan Rejected / High Risk")
            st.caption("Applicant exceeds acceptable risk parameters.")

    with res_col2:
        st.metric(
            label="Approval Confidence", 
            value=f"{prob:.1%}",
            delta="Low Risk" if prediction == 1 else "-High Risk",
            delta_color="normal" if prediction == 1 else "inverse"
        )

    # Probability visual bar
    st.write("**Approval Probability Gauge:**")
    st.progress(float(prob))