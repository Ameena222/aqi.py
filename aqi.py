import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title of the web app
st.title('Air Quality Index Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with air quality data", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the data
    st.write("Data preview:", df.head())

    # Check for missing values
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    # Data preprocessing
    # Assuming that the target column is 'AQI' and all other columns are features
    if 'AQI' in df.columns:
        X = df.drop(columns=['AQI'])
        y = df['AQI']

        # Handle any missing values (you can choose different imputation strategies here)
        X = X.fillna(X.mean())
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = model.predict(X_test)

        # Display the performance metrics
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # Display feature importances
        feature_importances = model.feature_importances_
        feature_names = X.columns
        feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        st.write("Feature Importances:", feature_df.sort_values(by='Importance', ascending=False))

        # Plot feature importance
        st.write("Feature Importances Plot:")
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel("Importance")
        plt.title("Feature Importances")
        st.pyplot(plt)

        # Prediction on new data
        st.subheader("Make a prediction")

        # Input fields for new data
        user_input = {}
        for feature in feature_names:
            user_input[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))

        # Make prediction for the entered data
        input_data = np.array([list(user_input.values())]).reshape(1, -1)
        prediction = model.predict(input_data)

        st.write(f"Predicted AQI: {prediction[0]:.2f}")

    else:
        st.error("The CSV file must contain an 'AQI' column.")
