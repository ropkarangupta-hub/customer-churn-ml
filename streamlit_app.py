import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('./models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('./models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Title and description
st.title('🎯 Customer Churn Prediction System')
st.markdown('### Predict whether a customer is likely to churn')

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader('Customer Demographics')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])
    
    st.subheader('Account Information')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', 
                                 ['Electronic check', 'Mailed check', 
                                  'Bank transfer (automatic)', 'Credit card (automatic)'])

with col2:
    st.subheader('Services')
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    
    st.subheader('Charges')
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=500.0)

# Create a button to make predictions
if st.button('Predict Churn', type='primary'):
    # Create input dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Preprocess the data (same as training)
    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_data)
    
    # Ensure all columns from training are present
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Select only the columns used in training
    input_encoded = input_encoded[feature_names]
    
    # Scale the features
    input_scaled = scaler.transform(input_encoded)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # Display results
    st.markdown('---')
    st.subheader('Prediction Results')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Prediction', 'CHURN' if prediction[0] == 1 else 'NO CHURN')
    
    with col2:
        st.metric('Churn Probability', f'{prediction_proba[0][1]:.2%}')
    
    with col3:
        st.metric('Retention Probability', f'{prediction_proba[0][0]:.2%}')
    
    # Add a visual indicator
    if prediction[0] == 1:
        st.error('⚠️ This customer is likely to churn. Consider retention strategies.')
    else:
        st.success('✅ This customer is likely to stay.')
    
    # Add a progress bar for visualization
    st.markdown('### Churn Risk Level')
    st.progress(float(prediction_proba[0][1]))

# Add footer
st.markdown('---')
st.markdown('*Customer Churn Prediction Model - Built with Streamlit*')