import streamlit as st
import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Assuming 'load_data' function is defined to load your housing data
def load_data():
    data_path = 'Housing.csv' 
    data = pd.read_csv('Housing.csv')
    return data
housing_data = load_data()

# Streamlit sidebar for filters
st.sidebar.header('Filters')

# Price Range Slider
price_min, price_max = st.sidebar.slider(
    "Price Range",
    float(housing_data['price'].min()),
    float(housing_data['price'].max()),
    (float(housing_data['price'].min()), float(housing_data['price'].max()))
)

# Bedroom Count Selector
bedroom_options = sorted(housing_data['bedrooms'].unique())
selected_bedrooms = st.sidebar.multiselect('Bedroom Count', bedroom_options, default=bedroom_options)

# Main Road Access Toggle
main_road_access = st.sidebar.checkbox('Main Road Access Only', False)

# Furnishing Status
furnishing_status_options = housing_data['furnishingstatus'].unique()
selected_furnishing_status = st.sidebar.multiselect('Furnishing Status', furnishing_status_options, default=furnishing_status_options)

# Filtering the data based on the selected filters
filtered_data = housing_data[
    (housing_data['price'] >= price_min) & (housing_data['price'] <= price_max) &
    (housing_data['bedrooms'].isin(selected_bedrooms)) &
    (housing_data['mainroad'] == 'yes') if main_road_access else housing_data['mainroad'].isin(['yes', 'no']) &
    (housing_data['furnishingstatus'].isin(selected_furnishing_status))
]

# Display filtered data - you can replace this with your visualization or data display code
st.write("Filtered Data", filtered_data)

# Example visualization with filters applied
st.subheader("Price Distribution After Applying Filters")
fig, ax = plt.subplots()
sns.histplot(filtered_data['price'], kde=True, ax=ax, color='skyblue')
ax.set_title('Filtered Price Distribution')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
st.pyplot(fig)
