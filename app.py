import streamlit as st

@st.cache_resource
def RNN_LSTM():
  '''
  TODO : Complete the GRU model
  Loads the RNN_LSTM Model based on a filepath
  '''
  try:
    # load model here
    pass
  except FileNotFoundError as e:
    print(f"Model file not found: {e}")
  return None

def GRU():
  '''
  TODO : Complete the GRU model
  Loads the RNN_LSTM Model based on a filepath
  '''
  try:
    # load model here
    pass
  except FileNotFoundError as e:
    print(f"Model file not found: {e}")
  return None

def simulate_inputs():
  '''
  TODO : Complete model training and adjust this as input
  Function that simulates sensor data readings
  '''
  return None

def main():
  '''
  TODO : Complete a simple dashboard that shows two parts
  • part of the RNN_LSTM
  • part of the GRU
  both parts is able to simulate a given input and returns the result of it.
  '''
  st.write("""# Poultry Defect Detection System""")
  modelA = RNN_LSTM()
  modelB = GRU()

if __name__ == "__main__":
  main()