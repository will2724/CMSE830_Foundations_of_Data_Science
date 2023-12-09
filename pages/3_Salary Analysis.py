
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

#removing data
df = df.loc[(df['Age'] >= 18) & (df['Age'] <= 120)]
df = df[df['Min. Salary'] > 30000]

###############################################################################################################################################
    

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

  

  #