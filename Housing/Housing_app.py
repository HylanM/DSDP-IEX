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

st.divider()

st.image("Pictures/iowa_loc.png")
st.caption("This shows where Ames is in Iowa")

st.write("""
***
## Overview of this project: 
The data set contains information on 2,930 properties in Ames, Iowa, 

including columns related to: house characteristics (bedrooms, garage, 

ireplace, pool, porch, etc.) location (neighborhood) lot information 

(zoning, shape, size, etc.)

""")

st.divider()

st.image("Pictures/skyline.png")
st.caption("This is the skyline of Ames")

st.write("""
## Objective:
The objective of this project is to develop a machine learning model 

that can accurately predict the sale prices of residential homes in Ames,

Iowa, by filtering and cleaning the data.The goal of this project is to take
         
the house characteristics and being able to describe how they impact the price of a house

""")