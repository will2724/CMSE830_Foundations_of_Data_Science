import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
iris_data = load_iris()
sns.pairplot(iris_data)
st.plotly_chart(iris_data, use_container_width=True)
