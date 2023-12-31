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
#df['age'] = [2023 - i  if i != -1 else i for i in df['Founded']]
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
df = df.loc[(df['Age'] >= 18) & (df['Age'] <= 120)]
df_notxt = df.drop(['Job Description'], axis=1)
mask = df.isin([-1]).any(axis=1)
df = df[~mask]
df['Title Simplified'] = df['Title Simplified'].replace('Na', 'NaN')
df = df[df['Title Simplified'].apply(lambda x: str(x).lower() != 'nan')]
df_sub = df

df['Revenue'] = df['Revenue'].replace('Unknown', 'NaN')
df = df[df['Revenue'].apply(lambda x: str(x).lower() != 'nan')]

df['Type of Ownership'] = df['Type of Ownership'].replace('Unknown', 'NaN')
df = df[df['Type of Ownership'].apply(lambda x: str(x).lower() != 'nan')]
df.reset_index(drop=True, inplace=True)

df_new = df.drop(['Salary Estimate', 'Job Description', 'Company Name', 'Size', 'Revenue',
       'Hourly', 'Employer Provided', 'company_txt', 'Seniority', 'Description Length'], axis=1)
df_stats = df_new.drop(['Python Exp.', 'AWS Exp.', 'Spark Exp.', 'Excel Exp.', 'Same State','Job Title', 'Location', 'HQ', 'Type of Ownership', 'Industry','Sector', 'Title Simplified' ], axis=1)

df = df[df['Avg. Salary'] > 30000]
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
st.set_page_config(page_title = 'Analysis of Data Scientist Openings',
                   page_icon = '💰',
                   layout='wide')

image = Image.open('1583217311227.png')
st.image(image, width=1200)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['**Description**', '**Simple Data Review**', '**Map Visualization**', '**Correlations**', '**Conclusion**'])

with tab1:
    st.title('Overview')
    st.write("""
    We long gone past the time when we once lived where every action required user input and the view on how math subjects can be applicable outside of what is required while in school.  Thanks to every growing field on Artificial Intelligence (AI) and Machine Learning (ML) and the well renowned [article published in Harvard Business Review](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century) back in 2012, which called Data Science the sexiest job of the 21st century. As businesses continue to seek ways effectively access trends (from present data, reviewing trends from past data and accurately predicting future trends), satisfy the need of customers while operating in the most efficient way possible.

    It is widely known that pursing a career in Data Science can be rewarding in terms of income, while that may be the case as the great saying goes 'More money, more problems'. This purpose of this app will explore through the job listings from a Glassdoor back in 2016 and observe certain trends in salaries, location, satisfaction of a position, the abundance of positions and see how they all relate to one another.""")
#    col1, col2 = st.columns(2,gap='large')
#    with col1.expander("Mo Money, Mo Problems"):
#        st.video("https://www.youtube.com/watch?v=NmowYxzKr6o",start_time=0)
#    with col2.expander("Benjamins"):
#        st.video("https://www.youtube.com/watch?v=n4p9zpEY6l8",start_time=0)
    st.write('#')

with tab2:
    st.title('Dataset Exploratation')
    col1, col2 = st.columns([4, 1])
    with col1:
        pursue_ds_career = st.radio('**How sure are you that you want to pursue a career in Data Science**', ['ehh maybe...', '**Very Certain❕**'], horizontal=True)
        if pursue_ds_career == 'ehh maybe...':
            st.write(df.head())
            st.write('#')
            st.subheader('Summary statistics of a ***few*** job postings')
            st.write(df.head().describe())
        else:
            st.write(df)
            st.write('#')
            st.subheader('Summary statistics of **all** job postings')
            st.write(df.describe())

    with col2:
        st.write('''
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

        **MLE: Machine Learning Engineer''')


    small_vis = st.radio('**Quick visuals**', ['Distributions', 'Coding Skillset'], horizontal=True)
    if small_vis == 'Distributions':
        st.write('''
        The message mentions that there is a bias in job openings, specifically in the context of Data Scientist positions and the type of ownership field being a private company. This suggests that there might be more job openings for Data Scientists in private companies compared to other roles. It can be seen that there are roughly 66% more Data Science job openings than the next option. This indicates that Data Scientist roles are abundant in the dataset.''')
        col1, col2 = st.columns(2)
        with col1:
            lst = ['Rating', 'Size', 'Founded', 'Age', 'Type of Ownership', 'Sector', 'Revenue', 'Title Simplified', 'Job State']
            #color_sel = st.sidebar.selectbox('Sorting Options', df.columns)
            st.sidebar.title('''Fig. 1A & 1B: Distribution of Features''')
            sel = st.sidebar.selectbox('Features', sorted(lst), index=7)

            if sel:
                color_mapping = {
                    value: color
                    for value, color in zip(df[sel].unique(), px.colors.qualitative.Set1)}
                df['Color'] = df[sel].map(color_mapping)
                fig = px.histogram(df, x=sel, color=sel)
                fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, title_text='Count'))
                st.plotly_chart(fig, use_container_width=True)
                st.title(f'Fig. 1A: Distribution for {sel}')
                with col2:
                    fig_pie = px.pie(df, names=sel, title=f'Pie chart for {sel}')
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie)
                    st.title(f'Fig. 1B: Percentage for {sel}')

    else:
        code_exp = df.groupby('Title Simplified')[['Python Exp.', 'Spark Exp.', 'AWS Exp.', 'Excel Exp.']].sum().drop_duplicates()
        st.sidebar.title('''Fig. 1B: Distribution of Coding Skillset''')
        code_sel = st.sidebar.selectbox('How does your coding experience help', code_exp.index)
        selected_data = code_exp.loc[code_sel]
        sorted_data = selected_data.sort_values()
        colors = ['crimson' if x == sorted_data.min() else 'green' if x == sorted_data.max() else 'lightslategray' for x in sorted_data]
        st.write('''
        Now you can observe the distributions which coding languages were idenitifed in the job listing for each of the simplified job titles. The data shows the experience in older languages, such as python and excel, are used more. Also, there are certain positions that require a knowledge in multiple skillsets.
        The column which is green identifies the coding language that was identified most often, column which is red/no color identifies the coding language that was identified most often.
        ''')
        fig = go.Figure(data=[go.Bar(
            x=sorted_data.index,
            y=sorted_data,
            marker_color=colors)])
        fig.update_layout(xaxis=dict(showgrid=False, title_text=f'Coding Experience for {code_sel}'), yaxis=dict(showgrid=False, title_text='Count'), title_text=f'Distribution of Coding Skillset Required for {code_sel} Position')
        st.plotly_chart(fig)
        st.title('''Fig. 2: Distribution of Coding Skillset''')

with tab3:
    col1, col2 = st.columns([5,1])
    #title_options = ['Fig. 2A: Average Salary per State','Fig. 2B: Average Job Satisfaction Rating per State','Fig. 2C: Total Number of Job Openings per State']
    st.sidebar.title('Fig. 3: Maps')
    #st.sidebar.title(f'{title_options[]}')
    option_box = st.sidebar.selectbox('Which map are you interested in viewing?',
    ['Salaries💸', 'Job Satisfaction 🎭', 'Opportunities 👩‍💻 🧑‍💻 ‍'])
    if option_box == 'Salaries💸':
        st.title('Fig. 3A: Average Salary per State')
        with col1:
            st.subheader("""Now let's view the average salary within this dataset to see where """)
            fig_salaries = px.choropleth(height = 1000, width = 1000,
            locations= df.groupby('Job State')['Avg. Salary'].mean().round(2).index,
            locationmode = 'USA-states',
            color = df.groupby('Job State')['Avg. Salary'].mean().round(2),
            color_continuous_scale = 'solar',
            labels = {'color':'Yearly Salary'})
            st.plotly_chart(fig_salaries.update_layout(geo_scope='usa'))
            with col2:
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('''Displayed is as a graph of the average salary in USD, per state, ranging from ~50k to ~135k''')

    elif option_box == 'Job Satisfaction 🎭':
        with col1:
            st.title('''Fig. 3B: Average Job Satisfaction Rating per State''')
            fig_rating = px.choropleth(height = 1100, width = 1100,
            locations = df.groupby('Job State')['Rating'].mean().index,
            locationmode = 'USA-states',
            color = round(df.groupby('Job State')['Rating'].mean(), 2),
            color_continuous_scale = 'solar',
            labels = {'color':'Employee Satisfaction Rating'})
            st.plotly_chart(fig_rating.update_layout(geo_scope='usa'))
            with col2:
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('''
                Displayed is the average satisfaction an employee in their position, this scale ranges from 1 to 5. (1 signifies complete dissatisfaction, 5 signifies absolute enjoyment in their position.)''')

    elif option_box == 'Opportunities 👩‍💻 🧑‍💻 ‍':
        st.title('''Fig. 3C: Total Number of Job Openings per State''')
        with col1:
            fig_states = px.choropleth(height = 1000, width = 1000,
            locations = df['Job State'].value_counts().index,
            locationmode = 'USA-states',
            color = df['Job State'].value_counts(),
            color_continuous_scale = 'solar',
            labels = {'color': 'Job Openings'})
            st.plotly_chart(fig_states.update_layout(geo_scope='usa'))
            with col2:
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('#')
                st.write('#')
                st.write(''' Displayed is the total count of job listings in each state.''')
#st.write("Let's review in the next tab how these 3 features relate to one another.")

with tab4:
    st.title('Correlation between Salary, Employee Satisfaction and their Location')
    job_count = df['Job State'].value_counts().reset_index()
    job_count.columns = ['Job State', 'Job Count']

    loc_avg_sal = df.groupby('Job State')[['Avg. Salary']].mean()
    loc_avg_sal.columns = ['Mean Avg Salary']

    loc_ratings = df.groupby('Job State')[['Rating']].mean()
    loc_ratings.columns = ['Mean Rating']

    df_tb = pd.merge(job_count, loc_avg_sal, on='Job State')
    df_tb = pd.merge(df_tb, loc_ratings, on='Job State')
    #df_tb = pd.merge(df_tb, df[['Job State', 'Title Simplified']], on='Job State')

    df_tb = df_tb.sort_values(by=['Mean Avg Salary', 'Mean Rating'], ascending=False)
    df_tb.reset_index(drop=True, inplace=True)
    df_tb_rating = df_tb[df_tb['Mean Rating'] >= 4.0]

    st.sidebar.title('Fig. 4: Correlations')
    slider = st.sidebar.select_slider('**Select:**', ['Top Salaies', 'Top Ratings','Best Overall'])
    col1, col2 = st.columns([4,2])

    #st.sidebar.subheader('''Change between the average salary by state and average employee satisfaction rating by state to view the best and worse state given the selected category.''')

    with col1:
        if slider == 'Top Salaies':
            df_tb = df_tb.sort_values(by='Mean Avg Salary', ascending=False).head().round(2)
            fig = px.scatter_3d(df_tb, x='Job Count', y='Mean Rating', z='Mean Avg Salary',
                                color='Job State', title='5 Highest Paying Average Salary by State')
            with col2:
                st.write(df_tb.head())
        elif slider == 'Top Ratings':
            df_tb = df_tb.sort_values(by='Mean Rating', ascending=False).head().round(2)
            fig = px.scatter_3d(df_tb, x='Job Count', y='Mean Rating', z='Mean Avg Salary',
                                color='Job State', title='5 Most Satisfied States Correlation')
            with col2:
                st.write(df_tb.head())
        else:
            df_tb_rating = df_tb_rating.round(2).head()
            fig = px.scatter_3d(df_tb_rating, x='Job Count', y='Mean Rating', z='Mean Avg Salary',
                                color='Job State', title='5 Most Satisfied States by State')
            with col2:
                st.write(df_tb_rating.head())

        fig.update_traces(marker=dict(size=20))
        fig.update_layout(width=900, height=900)
        st.plotly_chart(fig)


with tab5:
    st.title('Conclusion')
    st.write('''Throughout the exploration of this dataset certain factors have been analyzed to support your journey as you try to find the data science position that is right for you. It was also shown that the more money you make in a position equaites to having a higher level of happiness. This process included reviewal of underlying commonalities in the listings, an evaluation of different visualizations was made to help make this easier, including the comparing and contrasting of the relationship agamous Salary, Employee Satisfaction and their Location. I plan to further investigate this dataset by conducting testing and training of the data, this will allow for better accurate imputation of the missing data.''')









#
