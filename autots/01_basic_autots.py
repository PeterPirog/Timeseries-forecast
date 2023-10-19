from autots import AutoTS
from tools.timedata import ClimateTimeSeries

import warnings
warnings.filterwarnings('ignore')

# Load climate data
climate_data = ClimateTimeSeries(resample_frequency='1H', samples=24*31, selected_parameters=[1, 2, 5])
df = climate_data.dataframe
print(df)

# Configure and fit the AutoTS model
model = AutoTS(
    forecast_length=24,
    frequency='infer',
    prediction_interval=0.9,
    ensemble='auto',
    model_list="fast",
    transformer_list="fast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)

model = model.fit(df)

# Predict using the model
prediction = model.predict()

# Plot sample prediction
prediction.plot(model.df_wide_numeric, series=model.df_wide_numeric.columns[0])

# Print the details of the best model
print(model)

# Retrieve forecasts and their upper/lower bounds
forecasts_df = prediction.forecast
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# Get accuracy results
model_results = model.results()
validation_results = model.results("validation")
