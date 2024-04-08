import streamlit as st
st.header("Scalers")

#Import Info
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#Robust
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
#MinMax
from sklearn.preprocessing import MinMaxScaler
#QuantileTransformer
from sklearn.preprocessing import QuantileTransformer

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
#It selects the relevant features for the prediction and the target variable.
features = passengers[['Age', 'FirstClass', 'SecondClass', 'Sex_binary']]
target = passengers['Survived']

st.subheader("RobustScaler") 

scaler = RobustScaler()

# Without this I get an error saying that "NameError: name 'Y_train' is not defined"
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert y_train and y_test to NumPy arrays before reshaping
y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))  
y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)) 

# Creating the logistic regression model
model = LogisticRegression()

# Training the model with the training data
model.fit(X_train_scaled, y_train)

# You can now use model.predict(X_test_scaled) to make predictions on the test set
# And evaluate the model's performance
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f'Model Accuracy: {accuracy:.2f}')

st.subheader("MixMaxScaler") 
# Without this I get an error saying that "NameError: name 'Y_train' is not defined"
X_train, X_test, Y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

# Without this I get an error saying that "NameError: name 'Y_train' is not defined"
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert y_train and y_test to NumPy arrays before reshaping
y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))  
y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)) 

# Creating the logistic regression model
model = LogisticRegression()

# Training the model with the training data
model.fit(X_train_scaled, y_train)

# You can now use model.predict(X_test_scaled) to make predictions on the test set
# And evaluate the model's performance
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f'Model Accuracy: {accuracy:.2f}') 

st.subheader("QuantileTransformer") 

scaler = QuantileTransformer()

scaler = QuantileTransformer(n_quantiles=100)

X_train, X_test, Y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert y_train and y_test to NumPy arrays before reshaping
y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))  
y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)) 

# Creating the logistic regression model
model = LogisticRegression()

# Training the model with the training data
model.fit(X_train_scaled, y_train)

# You can now use model.predict(X_test_scaled) to make predictions on the test set
# And evaluate the model's performance
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f'Model Accuracy: {accuracy:.2f}')