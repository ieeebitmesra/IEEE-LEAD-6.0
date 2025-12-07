import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Model ---
# Load our new, feature-engineered model
try:
    model = joblib.load('final_model_v2.joblib')
except FileNotFoundError:
    st.error("Model file 'final_model_v2.joblib' not found. Please upload it to the GitHub repo.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Feature Engineering Function ---
# This is the *exact same function* we used in Colab.
def create_new_features(df):
    data = df.copy() 
    # Basic check to ensure columns exist
    if 'AveRooms' in data.columns and 'AveOccup' in data.columns:
        data['Rooms_per_person'] = data['AveRooms'] / data['AveOccup']
    if 'AveBedrms' in data.columns and 'AveRooms' in data.columns:
        data['Bedrms_per_room'] = data['AveBedrms'] / data['AveRooms']
    
    # Handle potential division by zero if AveRooms or AveOccup is 0
    data.replace([float('inf'), float('-inf')], 0, inplace=True)
    data.fillna(0, inplace=True) # Fill NaNs just in case
    return data

# --- 3. App Layout ---
st.set_page_config(page_title="California Housing Price Predictor", layout="wide")
st.title('🏡 California Housing Price Predictor (v2)')
st.write('This app predicts the median house value in a California district using a feature-engineered model.')
st.write('---')

# Create two columns for layout
col1, col2 = st.columns(2)

# Column 1: Sliders for the *original 8 features*
with col1:
    st.header('District Input Features')
    
    med_inc = st.slider('Median Income ($10,000s)', min_value=0.0, max_value=16.0, value=3.5, step=0.1)
    house_age = st.slider('House Age (years)', min_value=1, max_value=55, value=25, step=1)
    ave_rooms = st.slider('Average Rooms', min_value=1.0, max_value=15.0, value=5.0, step=0.1)
    ave_bedrms = st.slider('Average Bedrooms', min_value=1.0, max_value=10.0, value=1.0, step=0.1)
    population = st.slider('Population', min_value=1, max_value=40000, value=1400, step=10)
    ave_occup = st.slider('Average Occupancy', min_value=1.0, max_value=15.0, value=3.0, step=0.1)
    latitude = st.slider('Latitude', min_value=32.0, max_value=42.0, value=35.6, step=0.1)
    longitude = st.slider('Longitude', min_value=-125.0, max_value=-114.0, value=-119.5, step=0.1)

# Column 2: Display prediction
with col2:
    st.header('Prediction')
    
    if st.button('Predict Price', type="primary"):
        
        # --- 4. Prediction Logic ---
        
        # A. Create a DataFrame from the 8 slider inputs
        input_data = pd.DataFrame({
            'MedInc': [med_inc],
            'HouseAge': [house_age],
            'AveRooms': [ave_rooms],
            'AveBedrms': [ave_bedrms],
            'Population': [population],
            'AveOccup': [ave_occup],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })
        
        # B. Apply the feature engineering to the 8 inputs
        engineered_data = create_new_features(input_data)
        
        # C. Make a prediction using the *engineered data*
        try:
            prediction = model.predict(engineered_data)[0]
            
            # Format the prediction as currency
            prediction_price = prediction * 100000 
            
            # Display the result
            st.success(f'**Predicted Median House Value:**')
            st.header(f'${prediction_price:,.2f}')

            st.markdown("---")
            st.subheader("Features Used for Prediction:")
            st.write("This model uses the 8 features you provided, plus engineered features created from them (`Rooms_per_person` and `Bedrms_per_room`) to improve accuracy.")
            st.dataframe(engineered_data)
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("This may be due to a mismatch between the app's engineered features and the model's training data. Ensure all columns are present and in the correct order.")
