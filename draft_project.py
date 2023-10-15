import numpy as np
import pandas as pd
import re
import seaborn as sns
import streamlit as st
import os

#titles
st.title('Draft')
st.header('IDA')

my_dataset = 'data.csv'
#load dataset
@st.cache(persist=True)
def explore_data(data):
    df = pd.read_csv(os.path.join(my_dataset))
    return df



if st.checkbox('Preview Data'):
    data = explore_data(my_dataset)
    st.table(data.head())

preview = st.checkbox("Preview Data")
if preview:
    st.table(data.head())

df_shape = st.radio('What is the dimension of:', ('Entire Dataset', 'Rows', 'Columns'))
if data_shape == 'Entire':
    st.text('Entire Dataset Shown')
    st.write(data.shape)
elif data_shape == 'Rows':
    st.text('Rows Shown')
    st.write(data.shape[0])
else:
    st.text('Columns Shown')
    st.write(data.shape[1])

nan_in_ds = st.button('Are there any NaNs in dataset?')
if nan_in_ds:
    st.write(data.isna().any())
