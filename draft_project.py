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
    df.rename(columns = {'Type of ownership' : 'Type of Ownership',
                          'min_salary' : 'Min. Salary',
                          'max_salary' : 'Max. Salary',
                          'avg_salary' : 'Avg. Salary',
                          'job_state' : 'Job State',
                          'same_state' : 'Same State',
                          'age' : 'Age',
                          'python_yn' : 'Python Exp',
                          'R_yn' : 'R Exp.',
                          'spark' : 'Spark Exp.',
                          'aws' : 'AWS Exp.',
                          'excel' : 'Excel Exp.',
                          'job_simp' : 'Title Simplified',
                          'Headquarters' : 'HQ',
                          'hourly' : 'Hourly',
                          'desc_len' : 'Description Length',
                          'num_comp' : '# of Competitors',
                          'employer_provided' : 'Employer Provided',
                          'seniority' : 'Seniority',
                         })
    #df['Title Simplified'].str.replace('Mle', 'MLE')
    #df['Min. Salary']*1000
    df['Max. Salary'] = df['Max. Salary']*1000
    #df['Avg. Salary'] = df['Avg. Salary']*1000
    return df



if st.checkbox('Preview Data'):
    data = explore_data(my_dataset)
    st.table(data.head())

data_shape = st.radio('What is the dimension of:', ('Entire Dataset', 'Rows', 'Columns'))
#if data_shape == 'Entire Dataset':
#    st.text('Entire Dataset Shown')
#    st.write(data.shape)
#elif data_shape == 'Rows':
#    st.text('Rows Shown')
#    st.write(data.shape[0])
#else:
#    st.text('Columns Shown')
#    st.write(data.shape[1])

#if st.button('Are there any NaNs in dataset?'):
#    data = explore_data(my_dataset)
#    st.write(data.isna().any())
