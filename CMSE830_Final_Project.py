import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv')

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
df.replace({"Na": np.nan, "NaN": np.nan, "-1": np.nan, "Unknown / Non-Applicable": np.nan}, inplace=True)
df = df.drop(['Competitors', '# of Competitors'], axis=1)
df_orig = df

#removing data

df = df.loc[(df['Age'] >= 18) & (df['Age'] <= 120)]

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

####################################################################################################################################################################################################################################################################################################################################


st.set_page_config(page_title = 'ML',
                   page_icon = '🤖',
                   layout='wide')

#image = Image.open('istock-962219860-2-scaled.png')
#st.image(image, width=1200)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['**Description**', '**Simple Data Review**', '**Map Visualization**', '**Correlations**', '**Conclusion**'])

with tab1:
    st.title('Overview')
    st.write("""""")


with tab2:

    st.title('Dataset Exploratation')
    col1, col2 = st.columns([4, 2])
    with col1:
        with st.expander('**View Data**'):
            progression = st.radio('Progression in Exploration in Dataset', ['Original Dataset', 'Highlighting Missing Values', 'Removal Missing Values'], horizontal=True)
            if progression == 'Original Dataset':
                st.write(df_orig)
                with col2:
                    with st.expander('**View Data**'):
                        st.subheader('Summary Statistics of Job Postings')
                        st.write(df_orig.describe())
            elif progression == 'Highlighting Missing Values':
                def highlight_unknown(value, column_name, age):
                    if column_name == 'Revenue' and value == 'Unknown / Non-Applicable':
                        return 'background-color: red'
                    elif column_name == 'Title Simplified' and (value == 'Na' or value == 'Unknown / Non-Applicable'):
                        return 'background-color: red'
                    elif column_name == 'Age' and not (18 <= age <= 85):
                        return 'background-color: red'
                    else:
                        return ''
                df_highlighted = df_orig.style.applymap(lambda x: highlight_unknown(x, 'Revenue', 0), subset=['Revenue'])
                df_highlighted = df_highlighted.applymap(lambda x: highlight_unknown(x, 'Title Simplified', 0), subset=['Title Simplified'])
                df_highlighted = df_highlighted.applymap(lambda x: highlight_unknown(x, 'Age', x), subset=['Age'])
                st.write(df_highlighted)
                with col2:
                    with st.expander('**View Data**'):
                        st.subheader('Summary Statistics of Job Postings')
                        st.write(df_orig.describe())
            else:
                st.write(df)
                with col2:
                    with st.expander('**View Data**'):
                        st.subheader('Summary Statistics of Job Postings')
                        st.write(df.describe())
        
        


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
                df_orig['Color'] = df[sel].map(color_mapping)
                fig = px.histogram(df_orig, x=sel, color=sel)
                fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, title_text='Count'))
                st.plotly_chart(fig, use_container_width=True)
                st.title(f'Fig. 1A: Distribution for {sel}')
                with col2:
                    fig_pie = px.pie(df_orig, names=sel, title=f'Pie chart for {sel}')
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie)
                    st.title(f'Fig. 1B: Percentage for {sel}')

    else:
        code_exp = df_orig.groupby('Title Simplified')[['Python Exp.', 'Spark Exp.', 'AWS Exp.', 'Excel Exp.']].sum().drop_duplicates()
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



















        #