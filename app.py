import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib

@st.cache_resource
def Load_LSTM():
  try:
    model = load_model('model/LSTM/LSTM_bridge_forecaster.keras')
    scaler_X = joblib.load('model/LSTM/LSTM_feature_scaler.pkl')
    scaler_y = joblib.load('model/LSTM/LSTM_target_scaler.pkl')
    return model, scaler_X, scaler_y
  except FileNotFoundError as e:
    print(f"Model file not found: {e}")
    return None

def Load_GRU():
  try:
    model = load_model('model/GRU/gru_bridge_forecaster.keras')
    scaler_X = joblib.load('model/GRU/feature_scaler.pkl')
    scaler_y = joblib.load('model/GRU/target_scaler.pkl')
    return model, scaler_X, scaler_y
  except FileNotFoundError as e:
    print(f"Model file not found: {e}")
    return None

def simulate_inputs(model, scaler_X, scaler_y, feature_path):
  '''
  Function that simulates sensor data readings
  for context: the dataset has... Sensors (27): Accelerometers placed at various locations on the bridge
  '''
  features = pd.read_csv(feature_path, index_col='Timestamp')

  latest_no_data = 60
  test_input_raw = features.head(latest_no_data)
  test_input_scaled = scaler_X.transform(test_input_raw)

  input_for_model = test_input_scaled.reshape(1, latest_no_data, test_input_scaled.shape[1])
  forecast_steps = 20
  current_window = input_for_model.copy()
  forecast_list = []

  for _ in range(forecast_steps):
    prediction_scaled = model.predict(current_window, verbose=0)
    forecast_list.append(prediction_scaled[0, 0])
    next_step_sensors = current_window[:, -1:, :]
    current_window = np.append(current_window[:, 1:, :], next_step_sensors, axis=1)

  forecast_array = np.array(forecast_list).reshape(-1, 1)
  final_forecast = scaler_y.inverse_transform(forecast_array)

  return final_forecast

def plot_forecast(final_forecast):
  fig, ax = plt.subplots(figsize=(10, 5))

  ax.plot(final_forecast, marker='o', linestyle='--')
  ax.set_title('Recursive Structural Health Forecast')
  ax.set_ylabel('Health Index')
  ax.set_xlabel('Steps into Future')
  ax.legend(['Forecast'])

  return fig

def main():
  '''
  TODO : Complete a simple dashboard that shows two parts
  • part of the RNN_LSTM
  • part of the GRU
  both parts is able to simulate a given input and returns the result of it.
  '''

  st.write("""# Bridge Health Forecasting Application""")
  st.write("Select a model and simulate structural health forecasting.")
  LSTM = Load_LSTM()
  GRU = Load_GRU()
  option = st.radio(
    "Use the Model to be applied:",
    ("LSTM", "GRU")
  )

  if st.button("Run Model"):
    st.info("Running model... please wait")

    if option == "LSTM" and LSTM:
      model, scaler_X, scaler_y = LSTM
      featurePath = "model/features.csv"

    elif option == "GRU" and GRU:
      model, scaler_X, scaler_y = GRU
      featurePath = "model/GRU_features.csv"

    else:
      st.error("Model not loaded properly.")
      return

    result = simulate_inputs(model, scaler_X, scaler_y, featurePath)
    st.subheader("Forecast Values")
    st.write(result.flatten())
    fig = plot_forecast(result)
    st.pyplot(fig)

    st.success("Done!")

  if st.button("Reset"):
    st.experimental_rerun()

if __name__ == "__main__":
  main()