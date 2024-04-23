import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
import numpy as np

# Load your data
@st.cache  # This decorator helps to load the data only once and reuse it, which is useful for improving app performance
def load_data():
    data_path = 'Housing.csv' 
    data = pd.read_csv('Housing.csv')
    return data

housing_data = load_data()

# Streamlit app title
st.header("Housing Data Visualizations")

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Scale features
])

# Apply preprocessing pipeline to numerical features
numerical_features = ['area', 'price']
housing_data[numerical_features] = preprocessing_pipeline.fit_transform(housing_data[numerical_features])

# Setting the aesthetic style of the plots
sns.set_theme(style="whitegrid")

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

# Cross-validation
st.subheader("Cross-Validation")
X = housing_data.drop(columns=['price'])  # Features
y = housing_data['price']  # Target variable

model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
st.write("Cross-Validation Scores:", cv_scores)
st.write("Mean CV Score:", cv_scores.mean())

# Set up data for validation curve
X = housing_data.drop(columns=['price'])  # Features
y = housing_data['price']  # Target variable
alphas = np.logspace(-3, 3, 7)  # Varying values for regularization strength

# Create validation curve
train_scores, valid_scores = validation_curve(
    Ridge(), X, y, param_name="alpha", param_range=alphas, cv=5
)

# Plot validation curve
plt.figure()
plt.semilogx(alphas, np.mean(train_scores, axis=1), label='Training score', color='blue')
plt.semilogx(alphas, np.mean(valid_scores, axis=1), label='Validation score', color='red')
plt.fill_between(alphas, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.2, color='blue')
plt.fill_between(alphas, np.mean(valid_scores, axis=1) - np.std(valid_scores, axis=1),
                 np.mean(valid_scores, axis=1) + np.std(valid_scores, axis=1), alpha=0.2, color='red')
plt.title('Validation Curve')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
st.pyplot(plt)