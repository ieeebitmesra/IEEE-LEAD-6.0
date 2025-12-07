# California Housing Price Predictor

This project predicts the median house value in California districts using 1990 census data. It utilizes a Random Forest Regressor with custom feature engineering (such as rooms-per-person ratios) to overcome certain data limitations and achieve high accuracy.

### Live Demo
Click here to run the application: https://california-housing-predictions-2hxs9vadn4sqxpudiopkbt.streamlit.app/

## Files
- `housing_project.ipynb`: The main notebook containing EDA, feature engineering, and model training steps.
- `app_v2.py`: The deployment script for the Streamlit web application.
- `requirements.txt`: List of Python dependencies.
- `final_model_v2.joblib`: The trained and tuned Random Forest model (compressed).

## Results
- **RMSE:** 0.505
- **R² Score:** 0.804
- **Key Insight:** The model identified a data cap at $500,000, which was the primary source of error.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_v2.py
