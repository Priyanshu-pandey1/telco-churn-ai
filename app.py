import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the "Brain" and the "Translator"
# Make sure these files are in the same folder as app.py
model = pickle.load(open('churn_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# 2. Page Setup
st.set_page_config(page_title="Telco Churn AI", page_icon="📡")
st.title("📡 Customer Churn Prediction AI")
st.markdown("---")

# 3. Sidebar with your Professional Stats (The points I remembered for you!)
st.sidebar.header("Model Performance")
st.sidebar.write(f"**ROC-AUC Score:** 0.8419")
st.sidebar.write(f"**Test Accuracy:** 73.35%")
st.sidebar.write(f"**Generalization Gap:** 1.3%")
st.sidebar.write("---")
st.sidebar.info("This model uses an optimized XGBoost algorithm to identify high-risk customers.")

# 4. Organizing Inputs into Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ['Female', 'Male'])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.subheader("Services & Billing")
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
    payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# 5. Prediction Logic
if st.button("Predict Churn Risk"):
    # Create a dictionary for all inputs (matching the 19 columns the model expects)
    # Note: We are using placeholders for columns we didn't add to the UI for simplicity
    data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': internet,
        'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
        'PaymentMethod': payment, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([data])
    
    # Encode categorical features
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = encoder.fit_transform(input_df[col])
        
    # Get prediction
    prob = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    st.subheader("Results")
    if prob > 0.5:
        st.error(f"**HIGH RISK:** The probability of churn is {prob*100:.2f}%")
    else:
        st.success(f"**LOW RISK:** The probability of churn is {prob*100:.2f}%")