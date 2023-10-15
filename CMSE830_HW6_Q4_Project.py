#####  Mac
#/Users/sharodwilliams/CMSE830_Foundations_in_Data_Science/project/data.csv

##### Windows
#C:\\Users\\will2724\\CMSE830 Foundations to Data Science\data.csv

#load libraries and dataset
import numpy as np
import pandas as pd
import re
import seaborn as sns
import streamlit as st
import os

st.title('Data Scientist Salaries')
#col1, col2, col3, col4, col5  = st.columns([1,1,1,1,1.04])
#number = st.number_input('Enter your current salary:', min_value = 40000, value = 40000 )
#value = '125000'
#col5.metric(label = 'Average Data Scientist', value = '125000', delta = int(value) - int(number))
st.header('IDA')

def explore_data(dataset):
    df = pd.read_csv(os.path.join(dataset))
    return df
#df = pd.read_csv('/Users/sharodwilliams/CMSE830_Foundations_in_Data_Science/project/data.csv')
#df = pd.read_csv('https://github.com/will2724/CMSE830_Foundations_of_Data_Science/blob/main/data.csv')

#def load_data(file_name):
#    df = pd.read_csv(file_name)
#    return df


#df.pop('Unnamed: 0')
#age = [2023 - i  if i != -1 else i for i in df['Founded']]
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

#Obtain dataset/show dataset
preview = st.checkbox("Preview Data")
if preview:
    st.table(df.head())

df_shape = st.radio('What is the dimension of:', ('Entire Dataset', 'Rows', 'Columns'))
if df_shape == 'Entire':
    st.text('Entire Dataset Shown')
    st.write(df.shape)
elif df_shape == 'Rows':
    st.text('Rows Shown')
    st.write(df.shape[0])
else:
    st.text('Columns Shown')
    st.write(df.shape[1])

nan_in_ds = st.button('Are there any NaNs in dataset?')
if nan_in_ds:
    st.write(df.isna().any())


#new df created for Statistics
#df_new = df.drop(['Unnamed: 0', 'Salary Estimate', 'Job Description', 'Company Name', 'Size', 'hourly', 'Revenue', 'Competitors',
#       'hourly', 'employer_provided', 'company_txt', 'seniority', 'desc_len', 'num_comp'], axis=1)


df_stats = df.drop(['Description Length', 'Same State', 'Employer Provided', 'Description Length', 'Competitors'], axis=1)

#titles = df_new['Title Simp'].unique()


st.header('Some Statistics')
if st.toggle('Show Quantiles and other Statistics'):
    st.write(df.describe())

#Plotting
st.header('EDA 📊 📈')

df_cols = df.columns
#option = st.selectbox('Choose a Graph.', df_cols)
#plot = sns.histplot(df[df_cols])
#st.pyplot(plot.get_figure())
#why is this causing streamlit to be constantly running?

df_mask = np.triu(np.ones_like(df.corr()))
df_stats_mask = np.triu(np.ones_like(df_stats.corr()))
heatmap_lg = sns.heatmap(df.corr(), mask = df_mask, annot=True, cmap="cubehelix")
heatmap_sm = sns.heatmap(df_stats.corr(), mask = df_stats_mask, annot=True, cmap="cubehelix")
pairplot = sns.pairplot(data=df_stats, corner=True, hue='Title Simplified')
graphs = [pairplot, heatmap_sm, heatmap_lg]
#
#@st.cache_data

#show heatmap correlation for big df when toggle, and small correlation when toggle is on

option2 = st.selectbox('Lets look at correlations.', ['Pairplot', 'Heatmap'], index=0, placeholder='Select correlation graph')
if option2 == 'Pairplot':
    pairplot = sns.pairplot(data=df_stats, corner=True, hue='Title Simplified')
    st.pyplot(pairplot.fig)
else:
    heatmap = sns.heatmap(df.corr(), mask=df_mask, annot=True, cmap="cubehelix")
    st.pyplot(heatmap_lg.figure)



if st.toggle('On'):
    st.write(heatmap_lg)
else:
    st.write(heatmap_sm)

#st.pyplot(graphs)




#
