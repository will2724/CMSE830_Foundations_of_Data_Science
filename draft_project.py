import numpy as np
import pandas as pd
import re
import seaborn as sns
import streamlit as st
import os

#titles
st.title('Data Scientist Salaries')
st.header('IDA')

my_dataset = 'data.csv'
#load dataset
@st.cache(persist=True)
def explore_data(dataset):
    df = pd.read_csv(os.path.join(dataset))
    return df



if st.checkbox('Preview Data')
    data = explore_data()
    st.table(df.head())
