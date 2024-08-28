import streamlit as st
import pandas as pd

# Sample data
df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})

# Streamlit app
st.title('My First Streamlit App')
st.write('This is some text')
st.dataframe(df)

# Add a slider
value = st.slider('Select a value', 0, 100, 50)
st.write('Selected value:', value)
