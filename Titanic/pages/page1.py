import streamlit as st
st.header("Regressions")
st.divider()

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#Lasso
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
#Random Forest
from sklearn.ensemble import RandomForestRegressor


# Load the passenger data
passengers = pd.read_csv('passengers.csv')
features = passengers[['Age', 'FirstClass', 'SecondClass', 'Sex_binary']]
target = passengers['Survived']
#splits the data into training and testing sets to evaluate the model's performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#Scaler Used
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Y_train_scaled = scaler.fit_transform(X_train)
Y_test_scaled = scaler.transform(X_test)

st.subheader("Lasso Regression")

X_train, X_test, Y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert y_train and y_test to NumPy arrays before reshaping
y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))  
y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)) 

# Instantiate the Lasso Regression model
lasso_model = Lasso()

# Fit the model to the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lasso_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Model Accuracy: {accuracy_score:.2f}')

st.subheader("Decision Tree")
# Instantiate the Decision Tree Regression model
tree_model = DecisionTreeRegressor()

# Fit the model to the training data
tree_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = tree_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Model Accuracy: {accuracy_score:.2f}')

st.subheader("Random Forest")
# Instantiate the Random Forest Regression model
forest_model = RandomForestRegressor()

# Fit the model to the training data
forest_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = forest_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Model Accuracy: {accuracy_score:.2f}')