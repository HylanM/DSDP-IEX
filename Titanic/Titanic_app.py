import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Titanic Dataset")

st.header("My Titanic App")
st.write("""
**Author:** Maggie Hylan

**Source:** [Kaggle](https://www.kaggle.com/c/titanic/data)
""")
st.divider()

st.image("Pictures/current_titanic.png")
st.caption("This picture is what the titanic currently looks like")
st.divider()

st.write("""
***
## Overview of this project: 
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard, 

resulting in the death of 1502 out of 2224 passengers and crew.While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

For this project, I used several different types of classification and visualization models to convey the data. 

""")
st.divider()

st.image("Pictures/jack_rose.png")
st.caption("Jack and Rose from the famous Titanic movie")
st.divider()

st.write("""
---
## Problem Statement

The goal of this project is to develop a predictive model that accurately identifies factors influencing passenger survival rates during the tragic sinking of the RMS Titanic. 

By analyzing historical passenger data, we seek to uncover patterns and relationships between individual characteristics 

(such as age, gender, socio-economic class, cabin location, etc.) and their likelihood of survival.

""")