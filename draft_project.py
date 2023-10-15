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
    st.table(df.head())
