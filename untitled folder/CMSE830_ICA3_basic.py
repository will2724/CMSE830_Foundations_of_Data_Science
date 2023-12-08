import streamlit as st
import seaborn as sns
import pandas as pd

WI_BC_dataset = pd.read_csv('/Users/sharodwilliams/CMSE830: Foundations in Data Science/ICA/data.csv')

WI_BC_dataset.pop('id')
WI_BC_dataset.pop('Unnamed: 32')

WI_BC_dataset_columns = WI_BC_dataset.columns

option = st.selectbox(
    'Which graph would you like to view?',
    WI_BC_dataset_columns)

plot = sns.histplot(WI_BC_dataset[option])

st.pyplot(plot.get_figure())
