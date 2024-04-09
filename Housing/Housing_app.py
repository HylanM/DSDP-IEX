import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Housing Project")

st.header("My Housing App")
st.write("""
**Author:** Maggie Hylan

**Source:** [Kaggle](https://www.kaggle.com/c/Housing/data)
""")

st.write("""
***
## Overview of this project: 
The data set contains information on 2,930 properties in Ames, Iowa, 

including columns related to: house characteristics (bedrooms, garage, 

ireplace, pool, porch, etc.) location (neighborhood) lot information 

(zoning, shape, size, etc.)

""")