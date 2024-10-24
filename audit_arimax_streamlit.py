
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Load data
data = pd.read_csv('audit_metrics.csv', parse_dates=['Date'], index_col='Date')

# Define a function to run ARIMAX model
def run_arimax(train_data, exog_train, exog_test, arima_order=(1, 1, 1)):
    model = ARIMA(train_data, order=arima_order, exog=exog_train)
    result = model.fit()
    return result.get_forecast(steps=len(exog_test), exog=exog_test)

# Streamlit App
st.title("Audit Metrics Forecasting")
st.write("This app forecasts audit-related expenses based on external factors like GDP growth and CPI index.")

# Get user input for GDP and CPI adjustments
gdp_adjustment = st.slider("Adjust GDP Growth (%)", -0.05, 0.1, 0.03, step=0.005)
cpi_adjustment = st.slider("Adjust CPI Index (%)", 0.0, 0.5, 0.2, step=0.01)

# Adjust the external regressors
data['GDP_Growth_Adjusted'] = data['GDP_Growth'] + gdp_adjustment
data['CPI_Index_Adjusted'] = data['CPI_Index'] + cpi_adjustment

# Split into train/test data
train_data = data.iloc[:-6]
test_data = data.iloc[-6:]

# Exogenous variables
exog_train = train_data[['GDP_Growth_Adjusted', 'CPI_Index_Adjusted']]
exog_test = test_data[['GDP_Growth_Adjusted', 'CPI_Index_Adjusted']]

# Run ARIMAX
forecast = run_arimax(train_data['Expenses'], exog_train, exog_test)
forecast_df = pd.DataFrame({
    'Date': test_data.index,
    'Actual': test_data['Expenses'],
    'Forecasted': forecast.predicted_mean
})

st.write(forecast_df)
