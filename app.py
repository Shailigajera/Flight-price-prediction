import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and column names
model = joblib.load('flight_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Title
st.title('Flight Price Prediction')

# Dropdowns for categorical features
airline = st.selectbox('Select Airline', ['SpiceJet', 'AirAsia', 'Vistara', 'Indigo', 'Air India'])
source_city = st.selectbox('Select Source City', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai'])
departure_time = st.selectbox('Select Departure Time', ['Morning', 'Evening', 'Night', 'Afternoon', 'Early_Morning'])
stops = st.selectbox('Select Number of Stops', ['zero', 'one', 'two_or_more'])
arrival_time = st.selectbox('Select Arrival Time', ['Morning', 'Evening', 'Night', 'Afternoon', 'Early_Morning'])
destination_city = st.selectbox('Select Destination City', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai'])
flight_class = st.selectbox('Select Class', ['Economy', 'Business'])

# Number inputs for numerical features
duration = st.number_input('Enter Duration (in hours)', min_value=0.0, max_value=24.0, step=0.1)
days_left = st.number_input('Enter Days Left for Departure', min_value=0, max_value=365, step=1)

# Predict button
if st.button('Predict Flight Price'):
    features = {
        'airline': airline,
        'source_city': source_city,
        'departure_time': departure_time,
        'stops': stops,
        'arrival_time': arrival_time,
        'destination_city': destination_city,
        'class': flight_class,
        'duration': duration,
        'days_left': days_left
    }
    
    # Create a DataFrame for the input features
    input_df = pd.DataFrame([features])
    
    # One-hot encode the input features
    input_df = pd.get_dummies(input_df)
    
    # Ensure the input data has the same columns as the training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Predict the flight price
    prediction = model.predict(input_df)
    
    # Display the predicted price
    st.write(f'Predicted Flight Price: ₹{prediction[0]:,.2f}')
