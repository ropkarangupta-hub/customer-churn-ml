import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess

# Load saved model
model = joblib.load('./models/churn_model.pkl')
scaler = joblib.load('./models/scaler.pkl')
feature_names = joblib.load('./models/feature_names.pkl')

# App title
st.title('🔮 Customer Churn Prediction App')
st.markdown("""
This app predicts whether a telecom customer will churn (leave the company).
Enter customer details below to get a prediction.
""")

# Sidebar for prediction mode
st.sidebar.header('Prediction Mode')
mode = st.sidebar.radio('Choose prediction mode:', ['Single Customer', 'Batch Upload'])

if mode == 'Single Customer':
    st.header('📋 Enter Customer Details')
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Demographics')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])
        
        st.subheader('Services')
        phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        
    with col2:
        st.subheader('Account Information')
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', 
                                      ['Electronic check', 'Mailed check', 
                                       'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.number_input('Monthly Charges ($)', 0, 150, 50)
        total_charges = st.number_input('Total Charges ($)', 0, 10000, 500)
        
        st.subheader('Additional Services')
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    
    # Predict button
    if st.button('🔍 Predict Churn', type='primary'):
        # Create DataFrame from inputs
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
        
        # Preprocess
        processed_data = preprocess(input_data, mode='Online')
        
        # Ensure all required features are present
        for col in feature_names:
            if col not in processed_data.columns:
                processed_data[col] = 0
        
        processed_data = processed_data[feature_names]
        
        # Scale numeric features
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        processed_data[numeric_cols] = scaler.transform(processed_data[numeric_cols])
        
        # Predict
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        
        # Display results
        st.markdown('---')
        st.header('🎯 Prediction Results')
        
        if prediction == 1:
            st.error('⚠️ **HIGH RISK**: This customer is likely to CHURN!')
            st.metric('Churn Probability', f'{probability:.1%}')
            
            st.markdown('### 💡 Recommended Actions:')
            st.markdown("""
            - 📞 **Contact customer immediately** with retention offer
            - 💰 **Offer discount** or upgrade to long-term contract
            - 🛠️ **Provide tech support** if service issues exist
            - 🎁 **Special promotion** on add-on services
            """)
        else:
            st.success('✅ **LOW RISK**: Customer is likely to STAY!')
            st.metric('Retention Probability', f'{probability:.1%}')
            
            st.markdown('### 💡 Engagement Strategy:')
            st.markdown("""
            - 🌟 **Thank loyal customer** for their business
            - 📧 **Send satisfaction survey** to maintain quality
            - 🎉 **Offer loyalty rewards** to encourage long-term stay
            - 📱 **Promote new services** for upselling opportunities
            """)
        
        # Show probability breakdown
        st.markdown('---')
        st.subheader('Probability Breakdown')
        prob_df = pd.DataFrame({
            'Outcome': ['Will Stay', 'Will Churn'],
            'Probability': [probability, probability]
        })
        st.bar_chart(prob_df.set_index('Outcome'))

else:  # Batch Upload
    st.header('📤 Upload Customer Data File')
    st.markdown('Upload a CSV file with customer data for batch predictions.')
    
    uploaded_file = st.file_uploader('Choose CSV file', type='csv')
    
    if uploaded_file is not None:
        # Load data
        batch_data = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Data Preview')
        st.dataframe(batch_data.head())
        
        if st.button('🔍 Predict Batch', type='primary'):
            # Preprocess
            processed_batch = preprocess(batch_data, mode='Batch')
            
            # Ensure all features present
            for col in feature_names:
                if col not in processed_batch.columns:
                    processed_batch[col] = 0
            
            processed_batch = processed_batch[feature_names]
            
            # Scale numeric features
            numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            processed_batch[numeric_cols] = scaler.transform(processed_batch[numeric_cols])
            
            # Predict
            predictions = model.predict(processed_batch)
            probabilities = model.predict_proba(processed_batch)[:, 1]
            
            # Add predictions to original data
            result_df = batch_data.copy()
            result_df['Churn_Prediction'] = ['Will Churn' if p == 1 else 'Will Stay' for p in predictions]
            result_df['Churn_Probability'] = probabilities
            
            # Display results
            st.subheader('🎯 Prediction Results')
            st.dataframe(result_df)
            
            # Summary statistics
            churn_count = (predictions == 1).sum()
            total_count = len(predictions)
            churn_rate = churn_count / total_count * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric('Total Customers', total_count)
            col2.metric('Predicted Churns', churn_count)
            col3.metric('Churn Rate', f'{churn_rate:.1f}%')
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label='📥 Download Predictions',
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv'
            )

# Sidebar info
st.sidebar.markdown('---')
st.sidebar.info("""
**About This App**

This machine learning model predicts customer churn
based on demographic, service, and account information.

**Model:** Random Forest Classifier  
**Accuracy:** ~78-82%  

**Key Features:** Contract type, Tenure, Monthly charges
""")