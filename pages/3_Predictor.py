import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle

warnings.filterwarnings("ignore")

# Define file paths for models and preprocessor
models_dir = 'models'
dt_model_path = os.path.join(models_dir, 'dt_model.pkl')
rf_model_path = os.path.join(models_dir, 'rf_model.pkl')
preprocessor_path = os.path.join(models_dir, 'pipeline_preprocessor.pkl')

# Load the preprocessor pipeline
with open(preprocessor_path, 'rb') as file:
    preprocessor = joblib.load(file)

# Load DecisionTree model
with open(dt_model_path, 'rb') as file:
    dt_model = joblib.load(file)

# Load RandomForest model
with open(rf_model_path, 'rb') as file:
    rf_model = joblib.load(file)

# Load the preprocessing pipeline using the correct file path
pipeline_file_path = 'models/pipeline_preprocessor.pkl'
with open(pipeline_file_path, 'rb') as file:
    pipeline_preprocessor = pickle.load(file)

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier



# Define the predict_online function
def predict():
    # Add image
    st.image("images/image.png", use_column_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header('Demographics')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])

    with col2:
        st.header('Services')
        phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No phone service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No phone service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No phone service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No phone service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No phone service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No phone service'])

    with col3:
        st.header('Payments')
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.number_input('Monthly Charges', min_value=0)
        total_charges = st.number_input('Total Charges', min_value=0)
        tenure = st.number_input('Tenure', min_value=0)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
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

        # Preprocess data
        preprocessed_data = preprocessor.transform(input_data)

        selected_model = st.session_state.get('model', 'DecisionTree')
        if selected_model == 'DecisionTree':
            model = dt_model
        else:
            model = rf_model

        prediction = model.predict_proba(preprocessed_data)
        churn_percentage = prediction[0][1] * 100
        st.success(f'Churn Percentage ({selected_model} Model): {churn_percentage:.2f}%')

        # Visualize churn risk
        st.subheader("Churn Risk Meter")
        colors = ['#8A2BE2', '#FFFF00', '#FFA500']  # Violet, Yellow, Orange
        thresholds = [20, 40]
        levels = ['Low Churn Risk', 'Medium Churn Risk', 'High Churn Risk']
        risk_level = np.digitize(churn_percentage, thresholds, right=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_percentage,
            title={'text': "Churn Risk"},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': colors[risk_level]},
                'steps': [
                    {'range': [0, thresholds[0]], 'color': colors[0]},
                    {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                    {'range': [thresholds[1], 100], 'color': colors[2]}
                ],
            }
        ))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

        st.write(f"Churn Percentage: {churn_percentage:.2f}%")

def main():
    st.title("Churn Prediction Application")
    prediction_option = st.radio("Select Prediction Option", ["Online"])

    if prediction_option == "Online":
        st.session_state['model'] = st.selectbox('Select Model', ['DecisionTree', 'RandomForest'])
        predict()

    # elif prediction_option == "Batch":
    #     predict_batch()

if __name__ == '__main__':
    main()