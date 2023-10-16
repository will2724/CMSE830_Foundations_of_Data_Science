import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#titles
st.title('Draft')
st.header('IDA')


#load dataset
df = pd.read_csv('data.csv')
df.pop('Unnamed: 0')
df['age'] = [2023 - i  if i != -1 else i for i in df['Founded']]
df = df.rename(columns = {'Type of ownership' : 'Type of Ownership',
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
df['Title Simplified'] = df['Title Simplified'].str.replace('Mle', 'MLE')
df['Min. Salary'] = df['Min. Salary']*1000
df['Max. Salary'] = df['Max. Salary']*1000
df['Avg. Salary'] = df['Avg. Salary']*1000


if st.checkbox('Preview Data'):
    st.table(df.head())


data_shape = st.radio('What is the dimension of:', ('Entire Dataset', 'Rows', 'Columns'))
if data_shape == 'Entire Dataset':
    st.text('Entire Dataset Shown')
    st.write(df.shape)
elif data_shape == 'Rows':
    st.text('Rows Shown')
    st.write(df.shape[0])
else:
    st.text('Columns Shown')
    st.write(df.shape[1])

fig_map = st.radio("State-by-State,  What are you curious to explore?", ('Oppurtunities 👩‍💻 🧑‍💻 👨‍💻', 'Salaries💰 💳', 'Enjoyment 🎭'))
if fig_map == 'Oppurtunities 👩‍💻 🧑‍💻 👨‍💻':
    fig_states = px.choropleth(height = 800, width = 800,
        locations = df['Job State'].value_counts().index,
        locationmode = 'USA-states',
        color = df['Job State'].value_counts(),
        color_continuous_scale = 'balance',
        labels = {'color': 'Job Openings'},
        title = 'Jobs per State')
    plt.update_layout(geo_scope = 'usa')
    plt.show()
elif fig_map == 'Salaries💰 💳':
    fig_salaries = px.choropleth(height = 800, width = 800,
        locations= df.groupby('Job State')['Avg. Salary'].mean().index,
        locationmode = 'USA-states',
        color = round(df.groupby('Job State')['Avg. Salary'].mean(), 2),
        color_continuous_scale = 'balance',
        labels = {'color':'Yearly Salary'},
        title = 'Average Salary per State')
    plt.update_layout(geo_scope='usa')
    plt.show()
else:
    fig_rating = px.choropleth(height = 800, width = 800,
        locations = df.groupby('Job State')['Rating'].mean().index,
        locationmode = 'USA-states',
        color = round(df.groupby('Job State')['Rating'].mean(), 2),
        color_continuous_scale = 'balance',
        labels = {'color':'Employee Satisfaction Rating'},
        title = 'Employee Satisfaction Rating per State')
    plt.update_layout(geo_scope = 'usa')
    plt.show()

#if st.button('Are there any NaNs in dataset?'):
#    data = explore_data(my_dataset)
#    st.write(data.isna().any())
