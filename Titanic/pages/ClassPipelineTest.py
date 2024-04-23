import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load data
passengers = pd.read_csv('passengers.csv')

# Fill missing values
passengers['Age'].fillna(passengers['Age'].mean(), inplace=True)

# Define features and target
features = passengers[['Age', 'Pclass', 'Sex']]
target = passengers['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define preprocessing and modeling pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler()),  # Standardize features
    ('classifier', LogisticRegression())  # Logistic Regression model
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Streamlit app
st.header("Classifications")
st.subheader("Logistic Regression")
st.write(f'Model Accuracy: {accuracy:.2f}')

# Filters (remain unchanged)
st.write("""
---
## Filters

I wanted to add more to this page, so I thought that I would add ways to manipulate and 
filter the data. Being able to filter the data is essential for being able to understand 
which demographics suffered the most.
""")

# Your existing filters and data display code here...
data = pd.read_csv('passengers.csv')

# Passenger Class Filter
st.sidebar.header('Filter by Passenger Class')
# User can select multiple classes
selected_classes = st.sidebar.multiselect('Passenger Class', options=data['Pclass'].unique(), default=data['Pclass'].unique())

# Age Range Filter
st.sidebar.header('Filter by Age Range')
# User can select age range with a slider
min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
age_range = st.sidebar.slider('Age Range', min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Apply filters
filtered_data = data[(data['Pclass'].isin(selected_classes)) & (data['Age'] >= age_range[0]) & (data['Age'] <= age_range[1])]

# Display filtered data (you can replace this part with any visualization or data display)
st.header('Filtered Data Overview')
st.write(f"Number of passengers after applying filters: {len(filtered_data)}")
st.write(filtered_data)

# Example Visualization: Age Distribution of Filtered Data
st.header('Age Distribution of Filtered Passengers')
fig, ax = plt.subplots()
sns.histplot(filtered_data, x='Age', bins=20, kde=True, ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)
