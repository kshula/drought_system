import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# Load data from CSV into pandas DataFrame
@st.cache_data
def load_data():
    df = pd.read_csv('data\\data.csv')
    return df

# Preprocess data
def preprocess_data(df):
    # Separate features and target variable
    features = df.drop(['year', 'rainfall'], axis=1)
    target = df['rainfall']
    
    # One-hot encode the target variable 'rainfall'
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target_encoded, scaler

# Train models
def train_models(X, y):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model

    return trained_models

# Make predictions for future years
def make_predictions(models, scaler, forecast_years, df):
    predictions = {}

    # Forecast future features
    last_year = df['year'].max()
    future_features = []

    for year in range(1, forecast_years + 1):
        future_year = last_year + year
        # Use the last observed values for forecasting
        future_values = df.iloc[-1][1:-1]  # Exclude 'year' and 'rainfall'
        future_features.append(scaler.transform([future_values]))

    # Predictions for future years
    for idx, year in enumerate(range(1, forecast_years + 1)):
        future_year = last_year + year
        future_features_array = np.array(future_features[idx]).reshape(1, -1)
        
        for name, model in models.items():
            y_pred = model.predict(future_features_array)
            predicted_rainfall = y_pred[0]  # Assuming y_pred is the encoded rainfall category
            predictions.setdefault(f"Year {future_year}", {})[name] = predicted_rainfall

    return predictions

# Streamlit app
def main():
    st.title("Drought Prediction System with Machine Learning")

    # Load data and preprocess
    df = load_data()
    X, y, scaler = preprocess_data(df)

    # Train models
    models = train_models(X, y)

    # Main homepage
    st.header("Make Predictions for Future Years")
    forecast_years = st.slider("Select number of years to forecast ahead:", 1, 10, 5)

    if st.button("Predict"):
        st.write(f"Predictions for the next {forecast_years} years:")
        predictions = make_predictions(models, scaler, forecast_years, df)
        for year, models_predictions in predictions.items():
            st.write(year + ":")
            for model, prediction in models_predictions.items():
                st.write(f"{model}: {prediction}")

if __name__ == '__main__':
    main()
