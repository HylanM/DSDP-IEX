import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load your data
@st.cache  # This decorator helps to load the data only once and reuse it, which is useful for improving app performance
def load_data():
    data_path = 'Housing.csv' 
    data = pd.read_csv('Housing.csv')
    return data

housing_data = load_data()

# Setting Streamlit's page configuration (optional)
st.set_page_config(page_title="Housing Data Analysis", layout="wide")

# Streamlit app title
st.title("Housing Data Visualizations")

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Price Distribution
st.subheader("Price Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(housing_data['price'], kde=True, ax=ax1, color='skyblue')
ax1.set_title('Price Distribution')
ax1.set_xlabel('Price')
ax1.set_ylabel('Frequency')
st.pyplot(fig1)

# Area vs. Price Scatter Plot
st.subheader("Area vs. Price")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=housing_data, x='area', y='price', ax=ax2, alpha=0.6)
ax2.set_title('Area vs. Price')
ax2.set_xlabel('Area')
ax2.set_ylabel('Price')
st.pyplot(fig2)

# Count of Houses by Bedrooms
st.subheader("Count of Houses by Bedrooms")
fig3, ax3 = plt.subplots()
sns.countplot(data=housing_data, x='bedrooms', ax=ax3, palette='viridis')
ax3.set_title('Count of Houses by Bedrooms')
ax3.set_xlabel('Bedrooms')
ax3.set_ylabel('Count')
st.pyplot(fig3)

# Effect of Main Road Access on Price
st.subheader("Effect of Main Road Access on Price")
fig4, ax4 = plt.subplots()
sns.boxplot(data=housing_data, x='mainroad', y='price', ax=ax4, palette='coolwarm')
ax4.set_title('Effect of Main Road Access on Price')
ax4.set_xlabel('Main Road Access')
ax4.set_ylabel('Price')
st.pyplot(fig4)