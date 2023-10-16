import numpy as np
import pandas as pd
import re
import seaborn as sns
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go


@st.cache

# LOAD DATAFRAME FUNCTION
def load_data(path):
    df = pd.read_csv(path)
    return df

# LOAD GEIJASON FILE
with open("data/georef-switzerland-kanton.geojson") as response:
    geo = json.load(response)

# LOAD ENERGY DATA
df_orig = load_data(path="data/renewable_power_plants_CH.csv")
#df = deepcopy(df_orig)
