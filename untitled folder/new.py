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
from PIL import Image

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
df.pop('R Exp.')
df['Title Simplified'] = df['Title Simplified'].str.title()
df['Title Simplified'] = df['Title Simplified'].str.replace('Mle', 'MLE')
df['Min. Salary'] = df['Min. Salary']*1000
df['Max. Salary'] = df['Max. Salary']*1000
df['Avg. Salary'] = df['Avg. Salary']*1000

df_orig = df

#removing data
df = df.drop(['Competitors', '# of Competitors'], axis=1)
df = df.loc[(df['Age'] >= 16) & (df['Age'] <= 120)]
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
df_stats = df_new.drop(['Python Exp.', 'AWS Exp.', 'Spark Exp.', 'Excel Exp.', 'Same State','Job Title', 'Location', 'HQ', 'Type of Ownership', 'Industry','Sector', 'Title Simplified' ], axis=1)

#for col in df.columns:
#    unique_col[col] = df[col].unique()

scaler = MinMaxScaler()
sd_scaler = StandardScaler()
state_counts = df['Job State'].value_counts().reset_index()
state_counts.columns = ['Job State', 'Count']
state_counts['Normalized Count'] = scaler.fit_transform(state_counts[['Count']])
state_counts['Standardized Count'] = sd_scaler.fit_transform(state_counts[['Count']])

#df_stats =
#stat_cols = ['Rating', 'Founded', 'Min. Salary', 'Max. Salary', 'Avg_Salary']
#df_stats = df.drop(['Job Title', 'Salary Estimate', 'Job Description', 'Company Name', 'Location', 'HQ', 'Size', 'Type of Ownership', 'Industry', 'Sector', 'Revenue', 'Competitors',
#       'Hourly', 'Employer Provided', 'company_txt', 'Job State', 'Same State', 'Title Simplified', 'Seniority', '# of Competitors'], axis=1)
#df_stats_cols = df_stats.columns

####################################################################################################################################################################################################################################################################################################################################
st.set_page_config(page_title = 'Analysis of Data Sciectist Openings',
                   page_icon = '🧊',
                   layout='wide')

image = Image.open('1583217311227.png')
st.image(image, width=950)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['**Description**', '**Distributions**', '**Coding Languages**', '**Salaries**', '**Job Search**'])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Data Science is a multifaceted field that involves deriving valuable conclusions most often through the extrapolation from data. It integrates various domains and methodologies to interpret data, solve complex problems, and aid decision making. Despite in modern times, the debate as to whether the title of being called a Data Scientist is loosely used, the responsibilities usually don’t deviate far from covering crucial topics such as Exploratory Data Analysis (EDA), visualization, statistics, data wrangling, linear algebra, machine learning and optimization.")
        st.markdown("Data wrangling takes raw data and goes through the process of cleaning, rearranging, transforming, the removal of missing values from the data and generates a more comprehensive analysis through the use of multiple coding languages. The ability to aid viewers in understanding trends not only through words and numbers while interpreting your data is key reason why Data Visualization is vital conveying complex information in an intuitive way. Effective visualization enables easier understanding and interpretation of data patterns and trends.")
    with col2:
        st.header("Summary Statics of a dataset")
        st.write(df.describe())
        st.sidebar.write('''
        Founded: Year company was founded Job

        State: The state where the job is located

        Same State: An indicator of whether the job is in the same state as the person looking at the job

        Age: The age of the person looking at the job

        Python Exp.: An Indicator of whether the person looking at the job knows Python

        R Exp.: An indicator of whether the person looking at the job knows R

        Spark Exp.: An indicator of whether the person looking at the job knows Spark

        AWS Exp.: An Indicator of whether the person looking at the job knows AWS

        Excel Exp.: An indicator of whether the person looking at the job knows Excel

        Title Simplified: A simplified job title

        HQ: Location of Headquarters

        Hourly: An indicator of whether the person will be paid hourly

        Description Length: A count of total number of characters in the job posting


        **MLE: Machine Learning Engineer**
        ''')
        #if st.checkbox('Check for NaNs'):
        #    st.write(odf.isna().any())
    with col3:
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

    with tab2:
        lst = ['Rating', 'Size', 'Founded', 'Age', 'Type of Ownership', 'Sector', 'Revenue', 'Hourly', 'Title Simplified', 'Description Length', 'Job State']
        #color_sel = st.sidebar.selectbox('Sorting Options', df.columns)
        dist = st.selectbox('Distribution of Features', lst, index=1)
        if dist:
            sel = st.selectbox('Select sorting option', lst, index=1)
            color_mapping = {
                value: color
                for value, color in zip(df[sel].unique(), px.colors.qualitative.Set1)}
            df['Color'] = df[sel].map(color_mapping)
            fig = px.histogram(df, x=dist, color=sel)
            fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:

        #colors[1] = 'crimson'

        code_exp = df.groupby('Title Simplified')[['Python Exp.', 'Spark Exp.', 'AWS Exp.', 'Excel Exp.']].sum().drop_duplicates()
        code_sel = st.selectbox('How does your coding experience help', code_exp.index)
        selected_data = code_exp.loc[code_sel]
        sorted_data = selected_data.sort_values()
        colors = ['crimson' if x == sorted_data.min() else 'green' if x == sorted_data.max() else 'lightslategray' for x in sorted_data]

        fig = go.Figure(data=[go.Bar(
            x=sorted_data.index,
            y=sorted_data,
            marker_color=colors
        )])
        fig.update_layout(title_text=f'Coding Experience for {code_sel}')
        st.plotly_chart(fig)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Salary of an Average Data Sciectist")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x='Avg. Salary', ax=ax)
            st.pyplot(fig)
            st.write("But you will not become an average data sciectist right.....right?")
            if st.checkbox('Check box if you agree.'):
                with col2:
                    st.header('Spread of Salaies by Data Sciectists Job Titles')
                    fig, ax = plt.subplots(figsize=(8, 6))
                    fig = px.box(df, x='Avg. Salary', y= 'Title Simplified', color='Title Simplified', color_discrete_sequence= px.colors.qualitative.Dark24)
                    #sns.boxplot(data=df, x='Avg. Salary', y= 'Title Simplified', ax=ax)
                    st.plotly_chart(fig)
                    st.write("I'm glad you agree. Knowing you won't be any starndard Data Sciectist, listed above are ranges of the salaries, by job title.")

        option_box = st.selectbox('Choose',
        ['Salaries💰💳', 'Job Satisfaction 🎭'],
        index=0,
        placeholder="View Catageories"
        )
        col1, col2, col3, col4 = st.columns(4)
        if option_box == 'Salaries💰💳':
            with col1:
                fig_salaries = px.choropleth(height = 900, width = 900,
                locations= df.groupby('Job State')['Avg. Salary'].mean().index,
                locationmode = 'USA-states',
                color = round(df.groupby('Job State')['Avg. Salary'].mean(), 2),
                color_continuous_scale = 'bluered',
                labels = {'color':'Yearly Salary'},
                title = 'Average Salary per State')
                st.plotly_chart(fig_salaries.update_layout(geo_scope='usa'))
            with col4:
                st.write("#")
                st.write("#")
                st.write("#")
                st.title('''
                   ⬅️
                   Here we have
                ''')
        elif option_box == 'Job Satisfaction 🎭':
            with col1:
                fig_rating = px.choropleth(height = 1000, width = 1000,
                locations = df.groupby('Job State')['Rating'].mean().index,
                locationmode = 'USA-states',
                color = round(df.groupby('Job State')['Rating'].mean(), 2),
                color_continuous_scale = 'bluered',
                labels = {'color':'Satisfaction Rating'},
                title = 'Employee Satisfaction Rating per State')
                st.plotly_chart(fig_rating.update_layout(geo_scope='usa'))
            with col4:
                st.write("#")
                st.write("#")
                st.write("#")
                st.write("#")
                st.write("#")
                st.title('''
                   ⬅️
                   Here we have
                ''')

    with tab5:
        st.subheader('Now that you have explored differnt attributes of what it takes to become a Data Sciectist, either make a selection below to view all listings or based on your need, to select different features of a jobs listing to help find a Data Sciectist position.')
        st.write("#")

        if st.checkbox('View All Listings'):
            st.write(df)
            fig_states = px.choropleth(
                locations=state_counts['Job State'],
                locationmode='USA-states',
                color=state_counts['Standardized Count'],
                color_continuous_scale='bluered',
                labels={'color': '# of Job Openings per State'},
                title='Jobs per State')
            st.plotly_chart(fig_states.update_layout(geo_scope='usa'))

        st.write("#")

        job_title_sel = None
        sector_sel = None
        too_sel = None
        loc_sel = None
        min_sal_sel = None

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            job_title_sel = st.selectbox('Select your desired job title', df['Title Simplified'].unique()) # ,index=0, placeholder='Select')
        with col2:
            sector_sel = st.selectbox('Select your desired sector of work', df['Sector'].unique())
        with col3:
            too_sel = st.selectbox('Select your desired industry of work', df['Type of Ownership'].unique())
        with col4:
            st_sel = st.selectbox('Select your desired state to work', sorted(df['Job State'].unique()))
        with col5:
            min_sal_sel = st.number_input('Minimum Salary', min_value=df['Min. Salary'].min(), max_value=df['Min. Salary'].max())

        filtered_results = df[
        (df['Title Simplified'] == job_title_sel) &
        (df['Sector'] == sector_sel) &
        (df['Type of Ownership'] == too_sel) &
        (df['Job State'] == st_sel) &
        (df['Min. Salary'] >= min_sal_sel)]
        st.dataframe(filtered_results)













#
