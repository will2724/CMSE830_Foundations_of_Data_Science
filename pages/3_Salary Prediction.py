import streamlit as st
from PIL import Image
st.set_page_config(page_title = 'Analysis of Data Scientist Openings',
                   page_icon = '💰',
                   layout='wide')

image = Image.open('image2.png')
st.image(image, width=600)

##################################################################################################################################################################################################################################################################################

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',34)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
#from st_pages import Page, show_pages, add_page_title
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

df = df.loc[(df['Age'] >= 18) & (df['Age'] <= 120)]
df = df[df['Min. Salary'] > 30000]

df = df.dropna(axis=0)

############################################################################################################################################################################

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

# Preprocessing: One-hot encoding for categorical variables
categorical_features = ['Title Simplified', 'Size', 'Type of Ownership', 'Sector', 'Job State']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Creating a column transformer to apply transformations to the appropriate columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating the regression model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Training the model
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# Define the prediction function
def predict_salary(job_title, comp_size, ownership_type, sector, job_state):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Title Simplified': [job_title],
        'Size': [comp_size],
        'Type of Ownership': [ownership_type],
        'Sector': [sector],
        'Job State': [job_state]
    })

    # Making a prediction using the trained model
    predicted_salary = model.predict(input_data)
    
    return predicted_salary[0]

# Using the prediction function in Streamlit
if st.button("Predict Salary"):
    predicted_salary = predict_salary(job_title, comp_size, top, pick_sector, work_loc)
    st.write(f"Predicted Salary: ${predicted_salary:,.2f}")












