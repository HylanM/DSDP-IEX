import streamlit as st
import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
passengers['Age'].fillna(value=round(passengers['Age'].mean()), inplace=True)

passengers['FirstClass'] = passengers.Pclass.apply(lambda p: 1 if p == 1 else 0)
passengers['SecondClass'] = passengers.Pclass.apply(lambda p: 1 if p == 2 else 0)
passengers['Sex_binary'] = passengers.Sex.map({"male": 0, "female": 1})

features = passengers[['Age', 'FirstClass', 'SecondClass', 'Sex_binary']]
target = passengers['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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


# Preprocess data
passengers['Age'].fillna(passengers['Age'].mean(), inplace=True)
X = passengers[['Age', 'Pclass', 'Sex']]
y = passengers['Survived']

# Define pipeline
numeric_features = ['Age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

# Generate learning curve
@st.cache
def generate_learning_curve():
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipeline, X=X, y=y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy', n_jobs=-1)
    return train_sizes, train_scores, test_scores

train_sizes, train_scores, test_scores = generate_learning_curve()

# Plot learning curve
st.header("Learning Curve")
plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
st.pyplot(plt)