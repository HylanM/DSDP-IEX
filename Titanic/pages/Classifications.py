import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.header("Classifications")
st.divider()

st.write("""
---
## Disclaimer

I have learned a lot throughout this project. One of the things that I have learned is that 

the regressions that I have used are not necessarily helpful when trying to determine the
        
accuracy of a model. However, I still found them helpful in getting a better understanding of how 

regressions and classification work together and what they can do seperately.

""")

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

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
st.subheader("Logistic Regression")
st.write(f'Model Accuracy: {accuracy:.2f}')

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train_scaled, y_train)
y_pred = lasso_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
st.subheader("Lasso Regression")
st.write(f'Model Accuracy: {accuracy:.2f}')


# Decision Tree
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.subheader("Decision Tree")
st.write(f'Model Accuracy: {accuracy:.2f}')


# Random Forest
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
y_pred = forest_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.subheader("Random Forest")
st.write(f'Model Accuracy: {accuracy:.2f}')


