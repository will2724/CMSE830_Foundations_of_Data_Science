import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',34)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline

st.set_page_config(layout="wide")

data = pd.read_csv('data.csv')
data.pop('Unnamed: 0')
data = data.rename(columns = {'Type of ownership' : 'Type of Ownership',
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
data.pop('R Exp.')
data['Title Simplified'] = data['Title Simplified'].str.title()
data['Title Simplified'] = data['Title Simplified'].str.replace('Mle', 'MLE')
data['Min. Salary'] = data['Min. Salary']*1000
data['Max. Salary'] = data['Max. Salary']*1000
data['Avg. Salary'] = data['Avg. Salary']*1000
data.replace({"Na": np.nan, "NaN": np.nan, "-1": np.nan, "Unknown / Non-Applicable": np.nan}, inplace=True)
data = data.drop(['Competitors', '# of Competitors'], axis=1)
data['Type of Ownership'] = data['Type of Ownership'].replace('Unknown', 'Na')
data_orig = data
columns_to_drop = ['Seniority' , 'Job Title', 'Salary Estimate', 'Job Description', 'Company Name', 'Location', 'HQ', 'Industry', 'Hourly', 'Same State', 'Seniority', 'Employer Provided', 'company_txt']

df = data.drop(columns=columns_to_drop)
df_orig = df

#removing data
df = df.loc[(df['Age'] >= 18) & (df['Age'] <= 120)]
df = df[df['Min. Salary'] > 30000]

###############################################################################################################################################

from PIL import Image
image = Image.open('image2.png')

###############################################################################################################################################
###############################################################################################################################################
with st.sidebar: 
    selected = option_menu(
        menu_title="Menu",
        options=['Home', 'EDA', 'Salary Analysis', 'Salary Prediction', 'Conclusion','About Me'], 
        icons= ['house-door-fill' , 'rocket-takeoff-fill', 'graph-up', 'layers-fill', 'piggy-bank-fill' ]
    )

###############################################################################################################################################
###############################################################################################################################################
if selected == 'Home':
    st.image(image, width=800)
    st.write("""
    We long gone past the time when we once lived where every action required user input and the view on how math subjects can be applicable outside of what is required while in school.  Thanks to every growing field on Artificial Intelligence (AI) and Machine Learning (ML) and the well renowned [article published in Harvard Business Review](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century) back in 2012, which called Data Science (DS) the sexiest job of the 21st century. As businesses continue to seek ways effectively access trends (from present data, reviewing trends from past data and accurately predicting future trends), satisfy the need of customers while operating in the most efficient way possible.

    It is widely known that pursing a career in Data Science can be rewarding in terms of income, while that may be the case as the great saying goes 'More money, more problems'. This purpose of this app will explore through the job listings from a Glassdoor back in 2016 and observe certain trends in salaries, location, satisfaction of a position, the abundance of positions and see how they all relate to one another.""")

###############################################################################################################################################
###############################################################################################################################################
if selected == 'EDA':
    st.image(image, width=800)
    st.title('Exploratior Data Analysis')
    

col1, col2 = st.columns([4, 2])
with col1:
    with st.expander('**View Data**'):
        st.write(df)
        with col2:
            with st.expander('**View Data**'):
                st.subheader('Summary Statistics of Job Postings')
                st.write(df.describe())

tab1, tab2, tab3, tab4, tab5 = st.tabs(['**Description**', '**Simple Data Review**', '**Map Visualization**', '**Correlations**', '**Conclusion**'])

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
        select = st.checkbox('Show Description of Features')
        if select:
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
            st.sidebar.title('''Fig. 1A: Distribution of Features''')
            sel = st.sidebar.selectbox('Features', sorted(lst), index=7)

            if sel:
                color_mapping = {
                    value: color
                    for value, color in zip(df[sel].unique(), px.colors.qualitative.Set1)}
                df['Color'] = df[sel].map(color_mapping)
                fig = px.histogram(df, x=sel, color=sel)
                fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, title_text='Count'), title_text=f'Distribution for {sel}')
                st.plotly_chart(fig, use_container_width=True)
                st.title('Fig. 1A: Distribution of Features')
                with col2:
                    fig_pie = px.pie(df, names=sel, title=f'Pie chart for {sel}')
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie)
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
        st.title('''Fig. 1B: Distribution of Coding Skillset''')


###############################################################################################################################################
###############################################################################################################################################
if selected == 'Salary Analysis':
    st.image(image, width=800)
    st.title('Salary Analysis')
    
col1, col2 = st.columns([4,2])
with col1:
    num_feats = data.select_dtypes(include=['number']).columns
    num_feats = num_feats.drop(['Hourly', 'Employer Provided', 'Same State', 'Avg. Salary','Python Exp.', 'Spark Exp.', 'AWS Exp.', 'Excel Exp.'])
    number_of_feats = st.slider('Select the # of Features to Reflect Prediction', 1,3,1)
    df_cols = df.columns.drop(['Python Exp.', 'Spark Exp.', 'AWS Exp.', 'Excel Exp.'])
    available_feats =  num_feats.tolist()
    selected_feats = []

    for i in range(number_of_feats):
        if available_feats:
            feature_name = st.selectbox(f'Select Feature #{i + 1}', options=available_feats, key=i)
            selected_feats.append(feature_name)
            available_feats = [feat for feat in available_feats if feat != feature_name]
        else:
            st.warning('No more features available to select.')
            break  
with col2:
    code_skills = st.multiselect(
    'Select all coding languages you are familiar with:', 
    ['Python', 'Spark', 'AWS', 'Excel'], 
    placeholder="Choose an option")


y = df['Avg. Salary']
X = df[selected_feats]
test_fraction = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)


model_options = ["Linear Regression", "Ridge", "Lasso"]
selected_model = st.selectbox("Select a regression model:", model_options)


model_dict = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso()
}


model = model_dict[selected_model]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def evaluate_model (true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt (mean_squared_error (true, predicted))
    r2_square = r2_score (true, predicted)
    return mae, mse, rmse, r2_square
mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='markers', name='Actual'))
fig.add_trace(go.Scatter(x=np.arange(len(y_pred)), y=y_pred, mode='markers', name='Prediction'))
fig.update_layout(title="Actual vs Predicted Salary Values", xaxis_title="Index", yaxis_title="Values")
fig.update_traces(marker=dict(size=8,  symbol="diamond",
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
st.plotly_chart(fig)
st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
#st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
st.metric(label="R-Squared (R2)", value=f"{r2:.2f}")

###############################################################################################################################################
###############################################################################################################################################
if selected == 'Salary Prediction':
    st.title('')
    st.header("Let's Predict Your Salary")

#jt = data['Title Simplified'].tolsit()
st.subheader('What is simplified job tilie you are seeking?')

job_title = st.selectbox(
    "What job title best reflects your daily work?",
    ['Data Scientist', 'Analyst', 'Data Engineer', 'Director',
       'Manager', 'Machine Learning Engineer'])

comp_size = st.selectbox(
    "How many people are you hoping to have within your new company?",
    ['501 to 1000 employees', '10000+ employees',
       '1001 to 5000 employees', '201 to 500 employees',
       '5001 to 10000 employees', '51 to 200 employees',
       '1 to 50 employees'])

top = st.selectbox(
    "What type of ownership are you interested in?",
   ['Company - Private', 'Other Organization', 'Government',
       'Company - Public', 'Nonprofit Organization',
       'Subsidiary or Business Segment'])

pick_sector = st.selectbox(
    "Which sector are you interested in being apart of?",
   ['Aerospace & Defense', 'Health Care',
       'Oil, Gas, Energy & Utilities', 'Real Estate', 'Business Services',
       'Retail', 'Insurance', 'Transportation & Logistics', 'Finance',
       'Biotech & Pharmaceuticals', 'Telecommunications',
       'Information Technology', 'Manufacturing', 'Government',
       'Agriculture & Forestry', 'Education',
       'Arts, Entertainment & Recreation', 'Travel & Tourism', 'Media',
       'Non-Profit'])

work_loc   = st.selectbox(
    "Ideally what state would you work in?",
   ['NM', 'MD', 'WA', 'TX', 'VA', 'CO', 'KY', 'OR', 'MI', 'MA', 'CA',
       'NY', 'IL', 'AL', 'PA', 'GA', 'WI', 'NC', 'IN', 'MN', 'DC', 'OH',
       'TN', 'NJ', 'MO', 'ID', 'IA', 'FL', 'KS', 'UT', 'AZ'])

predictors = ['Title Simplified', 'Size', 'Type of Ownership', 'Sector', 'Job State']
target = 'Avg. Salary'
y = df[target]
X = df[predictors]
test_fraction = 0.2
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=42)

categorical_features = ['Title Simplified', 'Size', 'Type of Ownership', 'Sector', 'Job State']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


def predict_salary(job_title, comp_size, ownership_type, sector, job_state):
    input_data = pd.DataFrame({
        'Title Simplified': [job_title],
        'Size': [comp_size],
        'Type of Ownership': [ownership_type],
        'Sector': [sector],
        'Job State': [job_state]})
    predicted_salary = model.predict(input_data)
    return predicted_salary[0]

image2 = Image.open('IMG_0370.png')
if st.button("Predict Salary"):
    predicted_salary = predict_salary(job_title, comp_size, top, pick_sector, work_loc)
    st.write(f"Predicted Salary: ${predicted_salary:,.2f}")
    st.image(image2, width=100)

###############################################################################################################################################
###############################################################################################################################################
if selected == 'Conclusion':
    st.title('Conclusion')

###############################################################################################################################################
###############################################################################################################################################
if selected == 'About Me':
    st.title('Sharod Williams')
    
col1, col2 = st.columns([3,2])

with col1:
    image = Image.open('image2.png')
    st.image(image, width=600)
    st.title('Conclusion')
    st.write('''Throughout the exploration of this dataset certain factors have been analyzed to support your journey as you try to find the data science position that is right for you. It was also shown that the more money you make in a position equaites to having a higher level of happiness. This process included reviewal of underlying commonalities in the listings, an evaluation of different visualizations was made to help make this easier, including the comparing and contrasting of the relationship agamous Salary, Employee Satisfaction and their Location. I plan to further investigate this dataset by conducting testing and training of the data, this will allow for better accurate imputation of the missing data.''')
with col2:
    image2 = Image.open('IMG_0370.png')
    st.image(image2, width=300)
    st.write('Committed to using computational techniques to solve community problems, furthering my professional development through enrolling in courses and completing projects, while working independently and collaboratively as a data scientist/bioinformatician.')


#    


#st.set_page_config(page_title = 'Analysis of Data Scientist Openings', page_icon = '💰', layout='wide')



