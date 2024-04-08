import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore

st.header("Visualizations")


# Load the Titanic dataset
@st.cache
def load_data():
    data = pd.read_csv('passengers.csv')
    return data

data = load_data()

# Set the title of the app
st.title('Titanic Dataset Analysis')

# Overview Section
st.header('Dataset Overview')
st.write('This section provides a general overview of the Titanic dataset.')

# Display the shape of the dataset
st.write('Number of Rows:', data.shape[0])
st.write('Number of Columns:', data.shape[1])

# Data Table Section
st.header('Data Table')
st.write('Explore the Titanic dataset:')
st.dataframe(data)

# Visualizations Section
st.header('Visualizations')

# Histogram of Ages
st.subheader('Age Distribution')
fig, ax = plt.subplots()
data['Age'].dropna().plot(kind='hist', bins=20, ax=ax)
ax.set_xlabel('Age')
ax.set_ylabel('Count')
st.pyplot(fig)

# Pie Chart of Passenger Class Distribution
st.subheader('Passenger Class Distribution')
class_counts = data['Pclass'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig1)

# Survival Rate by Gender
st.subheader('Survival Rate by Gender')
survival_rate = data.groupby('Sex')['Survived'].mean()
fig2, ax2 = plt.subplots()
survival_rate.plot(kind='bar', ax=ax2)
ax2.set_ylabel('Survival Rate')
st.pyplot(fig2)

