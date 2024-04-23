import streamlit as st
import pandas as pd
#######ATTEMPT AT PIPELINE###########
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#######ATTEMPT AT PIPELINE###########
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_score

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



#######ATTEMPT AT PIPELINE###########
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
#######ATTEMPT AT PIPELINE###########

###__if the cross val didn't work try switching the variables x and y)
# Perform cross-validation
cv_scores = cross_val_score(pipeline, X_test_scaled, y_test, cv=5)  # 5-fold cross-validation
# Display results
st.header("Cross-Validation Results")
st.write("Mean Accuracy: {:.2f}".format(cv_scores.mean()))
st.write("Standard Deviation of Accuracy: {:.2f}".format(cv_scores.std()))

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
st.subheader("Logistic Regression")
st.write(f'Model Accuracy: {accuracy:.2f}')
# Report accuracy with standard deviation
mean_accuracy = cv_scores.mean()
std_accuracy = cv_scores.std()

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

st.write("""
---
## Filters

I wanted to add more to this page, so I thought that I would add ways to manipulate and 
filter the data. Being able to filter the data is essential for being able to understand 
which demographics suffered the most.
         
""")

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
