# import streamlit as st
# import pandas as pd

# uploaded_file = st.file_uploader("Choose a CSV file")
# if uploaded_file is not None:
#     print(uploaded_file)
#     df = pd.read_csv(uploaded_file)
#     st.write(df)

import streamlit as st
import pandas as pd

# Replace with your desired file path
# file_path = "/tmp/iris.csv"
# file_path = "iris.csv"
file_path = "../../data/samples/iris.csv"

try:
  df = pd.read_csv(file_path)
  st.write(df)
except FileNotFoundError:
  st.error("File not found.")

