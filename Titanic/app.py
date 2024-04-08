import streamlit as st
import pandas as pd
import numpy as np


st.header("My Titanic App")
st.write("""
**Author:** Maggie Hylan 

**Source:** [Kaggle](https://www.kaggle.com/c/titanic/data)
""")

##import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 

# Load the passenger data
passengers = pd.read_csv('passengers.csv')

# Fill the nan values in the age column  #DOBBIE
passengers['Age'].fillna(value = round(passengers['Age'].mean()), inplace = True)

# Create a first class column
passengers['FirstClass'] = passengers.Pclass.apply( lambda p: 1 if p == 1 else 0)

passengers['SecondClass'] = passengers.Pclass.apply( lambda p: 1 if p == 2 else 0)
passengers['Sex_binary'] = passengers.Sex.map({"male": 0, "female": 1})

#It selects the relevant features for the prediction and the target variable.
features = passengers[['Age', 'FirstClass', 'SecondClass', 'Sex_binary']]
target = passengers['Survived']

#It selects the relevant features for the prediction and the target variable.
features = passengers[['Age', 'FirstClass', 'SecondClass', 'Sex_binary']]
target = passengers['Survived']

#splits the data into training and testing sets to evaluate the model's performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#standardizes the features to have a mean of 0 and a standard deviation of 1,
#which is particularly important for logistic regression  to ensure all features contribute equally to 
#the prediction
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Y_train_scaled = scaler.fit_transform(X_train)
Y_test_scaled = scaler.transform(X_test)

# Creating the logistic regression model
model = LogisticRegression()

# Training the model with the training data
model.fit(X_train_scaled, y_train)

# You can now use model.predict(X_test_scaled) to make predictions on the test set
# And evaluate the model's performance
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f'Model Accuracy: {accuracy:.2f}')