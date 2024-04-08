import streamlit as st
import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

st.header("Visualizations")

st.write("""
---       
## List of Column Names and what the values represent
         
| Column Name    | Description                                                                     |
|----------------|---------------------------------------------------------------------------------|
| PassengerId    | A unique numerical identifier assigned to each passenger.                         |
| Survived       | Survival status of the passenger (0 = No, 1 = Yes).                              |
| Pclass         | The passenger's ticket class (1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class).   |
| Name           | The passenger's full name.                                                      |
| Sex            | The passenger's gender (male, female).                                         |
| Age            | The passenger's age in years. Fractional values may exist for younger children. |
| SibSp          | The number of siblings or spouses traveling with the passenger.                   |
| Parch          | The number of parents or children traveling with the passenger.                   |
| Ticket         | The passenger's ticket number.                                                  |
| Fare           | The price the passenger paid for their ticket.                                  |
| Cabin          | The passenger's cabin number (if recorded).                                    |
| Embarked       | The passenger's port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). |
---
""")

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


# Fare Distribution Plot
st.header('Fare Distribution')
fig, ax = plt.subplots()
sns.histplot(data=data, x='Fare', kde=True, ax=ax)
ax.set_title('Distribution of Fares Paid by Passengers')
st.pyplot(fig)

# Embarkation Points Plot
st.header('Passengers by Embarkation Point')
embarkation_counts = data['Embarked'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=embarkation_counts.index, y=embarkation_counts.values, ax=ax)
ax.set_title('Number of Passengers by Embarkation Point')
ax.set_xlabel('Embarkation Point')
ax.set_ylabel('Number of Passengers')
st.pyplot(fig)