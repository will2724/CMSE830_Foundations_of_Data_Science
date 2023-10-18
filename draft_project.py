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
#from PIL import Image

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
                      'python_yn' : 'Python Exp.',
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
df['Title Simplified'] = df['Title Simplified'].str.title()
df['Title Simplified'] = df['Title Simplified'].str.replace('Mle', 'MLE')
df['Min. Salary'] = df['Min. Salary']*1000
df['Max. Salary'] = df['Max. Salary']*1000
df['Avg. Salary'] = df['Avg. Salary']*1000

#removing data
df = df.drop(['Competitors', '# of Competitors'], axis=1)
df_notxt = df.drop(['Job Description'], axis=1)
mask = df.isin([-1]).any(axis=1)
df = df[~mask]
df['Title Simplified'] = df['Title Simplified'].replace('Na', 'NaN')
df['Revenue'] = df['Revenue'].replace('Unknown', 'NaN')
df['Type of Ownership'] = df['Type of Ownership'].replace('Unknown', 'NaN')
df = df[df['Title Simplified'].apply(lambda x: str(x).lower() != 'nan')]
df = df[df['Title Simplified'].apply(lambda x: str(x).lower() != 'nan')]
df = df[df['Type of Ownership'].apply(lambda x: str(x).lower() != 'nan')]
df.reset_index(drop=True, inplace=True)

df_new = df.drop(['Salary Estimate', 'Job Description', 'Company Name', 'Size', 'Revenue',
       'Hourly', 'Employer Provided', 'company_txt', 'Seniority', 'Description Length'], axis=1)
df_stats = df_new.drop(['Python Exp.', 'R Exp.', 'AWS Exp.', 'Spark Exp.', 'Excel Exp.', 'Same State','Job Title', 'Location', 'HQ', 'Type of Ownership', 'Industry','Sector', 'Title Simplified' ], axis=1)

df_cols = df.columns
unique_col = {}
for col in df.columns:
    unique_col[col] = df[col].unique()
code_exp = df.groupby('Title Simplified')[['Python Exp.', 'R Exp.', 'Spark Exp.', 'AWS Exp.', 'Excel Exp.']].sum().drop_duplicates()

#df_stats =
#stat_cols = ['Rating', 'Founded', 'Min. Salary', 'Max. Salary', 'Avg_Salary']
#df_stats = df.drop(['Job Title', 'Salary Estimate', 'Job Description', 'Company Name', 'Location', 'HQ', 'Size', 'Type of Ownership', 'Industry', 'Sector', 'Revenue', 'Competitors',
#       'Hourly', 'Employer Provided', 'company_txt', 'Job State', 'Same State', 'Title Simplified', 'Seniority', '# of Competitors'], axis=1)
#df_stats_cols = df_stats.columns
#======================================================================================

image = Image.open('1583217311227.png')


tab1, tab2 , tab3, tab4 = st.tabs(['Description', 'Required Skills', 'Salary', 'Job Openings'])
g = sns.PairGrid(df)
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, width=850)
        st.markdown("Data Science is a multifaceted field that involves deriving valuable conclusions most often through the extrapolation from data. It integrates various domains and methodologies to interpret data, solve complex problems, and aid decision making. Despite in modern times, the debate as to whether the title of being called a Data Scientist is loosely used, the responsibilities usually don’t deviate far from covering crucial topics such as Exploratory Data Analysis (EDA), visualization, statistics, data wrangling, linear algebra, machine learning and optimization.")
        st.markdown("Data wrangling takes raw data and goes through the process of cleaning, rearranging, transforming, the removal of missing values from the data and generates a more comprehensive analysis through the use of multiple coding languages. The ability to aid viewers in understanding trends not only through words and numbers while interpreting your data is key reason why Data Visualization is vital conveying complex information in an intuitive way. Effective visualization enables easier understanding and interpretation of data patterns and trends.")
    with col2:
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.header("Summary Statics of a dataset")
        st.write(df.describe())
        title = st.text_input('Example of data wrangling', "code_exp = df.groupby('Title Simplified')[['Python Exp.', 'R Exp.', 'Spark Exp.', 'AWS Exp.', 'Excel Exp.']].sum().drop_duplicates()", disabled=True)
        st.write('This is a real life example')

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
        if st.checkbox('Check for NaNs'):
            st.write(df.isna().any())
        #st.plotly(sns.pairplot(data=df_stats_mask, corner=True, hue='Title Simplified'))

with tab2:
    st.markdown("Data wrangling takes raw data and goes through the process of cleaning, rearranging, transforming, the removal of missing values from the data and generates a more comprehensive analysis through the use of multiple coding languages. The ability to aid viewers in understanding trends not only through words and numbers while interpreting your data is key reason why Data Visualization is vital conveying complex information in an intuitive way. Effective visualization enables easier understanding and interpretation of data patterns and trends. ")
    def comparison_plot(df, x_column, y_column):
        fig = px.bar(df, x= x_column, y=y_column, color='Title Simplified', barmode='group')
        fig.update_layout(title= f'{x_column} Comparison', xaxis_title=x_column, yaxis_title='Coding Skills')
        st.plotly_chart(fig)
    selected_x_column = st.sidebar.selectbox("Select X-Axis Column", df.columns,index=9,
   placeholder="Select contact method...")
    selected_y_column = st.sidebar.selectbox("Select Y-Axis Coding Skill", df.columns[22:27], index=0,
   placeholder="Select contact method...")


    if not selected_x_column or not selected_y_column:
        st.warning("Please select an X-Axis column and a Y-Axis coding skill.")
    else:
        comparison_plot(df, selected_x_column, selected_y_column)

    #multi_sel_cols = ['Job State','Title Simplified', 'Type of Ownership', 'Min. Salary', 'Max. Salary', 'Avg. Salary', 'Age', 'Industry', 'Sector']

    #def int_plot(df):
    #    x_vals = st.selectbox('1st Comparison', multi_sel_cols)
    #    y_vals = st.selectbox('2nd Comparison', multi_sel_cols)
    #    plot= px.bar(df, x_vals, y_vals)
        #st.plotly_chart(plot)

#int_plot(df)
#with tab3:
    #salary = st.checkbox("Salary")

# Sidebar text
#description = st.sidebar.text("You are viewing the Description tab")
#required_skills:
#    st.sidebar.text("You are viewing the Required Skills tab")
#salary:
#    st.sidebar.text("You are viewing the Salary tab")

with tab3:
    st.subheader("Salary of an Average Data Sciectist")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='Avg. Salary', ax=ax)
    st.pyplot(fig)
    st.write("But you will not become an average data sciectist right.....right?")
    if st.checkbox('RIGHT!'):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig = px.box(df, x='Avg. Salary', y= 'Title Simplified', color='Title Simplified', color_discrete_sequence= px.colors.qualitative.Dark24)
        #sns.boxplot(data=df, x='Avg. Salary', y= 'Title Simplified', ax=ax)
        st.plotly_chart(fig)
    option_box = st.selectbox('Choose',
    ['Salaries💰💳', 'Enjoyment 🎭'],
    index=0,
    placeholder="View Catageories"
    )
    if option_box == 'Salaries💰💳':
        fig_salaries = px.choropleth(height = 800, width = 800,
        locations= df.groupby('Job State')['Avg. Salary'].mean().index,
        locationmode = 'USA-states',
        color = round(df.groupby('Job State')['Avg. Salary'].mean(), 2),
        color_continuous_scale = 'balance',
        labels = {'color':'Yearly Salary'},
        title = 'Average Salary per State')
        st.plotly_chart(fig_salaries.update_layout(geo_scope='usa'))
    if option_box == 'Enjoyment 🎭':
        fig_rating = px.choropleth(height = 800, width = 800,
        locations = df.groupby('Job State')['Rating'].mean().index,
        locationmode = 'USA-states',
        color = round(df.groupby('Job State')['Rating'].mean(), 2),
        color_continuous_scale = 'balance',
        labels = {'color':'Employee Satisfaction Rating'},
        title = 'Employee Satisfaction Rating per State')
        st.plotly_chart(fig_rating.update_layout(geo_scope='usa'))
    #else:
with tab4:

    st.table(df_notxt.head())
    if st.checkbox('View All Listings'):
        st.write(df)
    st.subheader('View of Job Postings Across the US')
    fig_states = px.choropleth(height = 800, width = 800,
    locations = df['Job State'].value_counts().index,
    locationmode = 'USA-states',
    color = df['Job State'].value_counts(),
    color_continuous_scale = 'balance',
    labels = {'color': 'Job Openings'},
    title = 'Jobs per State')
    st.plotly_chart(fig_states.update_layout(geo_scope='usa'))
