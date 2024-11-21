import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the UI
st.title("Compressed Earth Brick Strength Predictor")
st.sidebar.header("Input Ratios")

# Input fields for material ratios
sand = st.sidebar.slider("Sand (%)", 0, 100, 50)
silt = st.sidebar.slider("Silt (%)", 0, 100, 25)
clay = st.sidebar.slider("Clay (%)", 0, 100, 20)
cement = st.sidebar.slider("Portland Cement (%)", 0, 30, 5)

# Placeholder for user data
input_data = pd.DataFrame({
    'Sand': [sand],
    'Silt': [silt],
    'Clay': [clay],
    'Cement': [cement]
})

# Optional inputs with checkboxes
use_compression_pressure = st.sidebar.checkbox("Specify Compression Pressure?", value=True)
compression_pressure = (
    st.sidebar.slider("Compression Pressure (MPa)", 0.0, 20.0, 10.0)
    if use_compression_pressure
    else np.nan
)

use_density = st.sidebar.checkbox("Specify Density?", value=True)
density = (
    st.sidebar.slider("Density (kg/m³)", 1500, 2500, 2000)
    if use_density
    else np.nan
)

use_thermal_conductivity = st.sidebar.checkbox("Specify Thermal Conductivity?", value=True)
thermal_conductivity = (
    st.sidebar.slider("Thermal Conductivity (W/m·K)", 0.1, 2.0, 0.5)
    if use_thermal_conductivity
    else np.nan
)

use_porosity = st.sidebar.checkbox("Specify Porosity?", value=True)
porosity = (
    st.sidebar.slider("Porosity (%)", 0, 50, 20)
    if use_porosity
    else np.nan
)

use_modulus = st.sidebar.checkbox("Specify Elastic Modulus?", value=True)
modulus = (
    st.sidebar.slider("Elastic Modulus (GPa)", 0.1, 50.0, 5.0)
    if use_modulus
    else np.nan
)

# Input data with missing values handled
input_data = pd.DataFrame({
    'Sand': [sand],
    'Silt': [silt],
    'Clay': [clay],
    'Cement': [cement],
    'Compression Pressure': [compression_pressure],
    'Density': [density],
    'Thermal Conductivity': [thermal_conductivity],
    'Porosity': [porosity],
    'Modulus': [modulus]
})

# Handle missing values by replacing with mean or using the actual column data
for column in input_data.columns:
    if input_data[column].isnull().values.any():
        mean_value = input_data[column].mean()  # Use the mean of the feature from the dataset
        input_data[column].fillna(mean_value, inplace=True)

st.write("### Input Parameters (After Handling Missing Values)")
st.write(input_data)

# Example dataset (replace this with experimental data)
@st.cache_data
def load_data():
    try:
        # Read CSV file and handle extra columns or whitespace issues
        data = pd.read_csv('CSEB.csv', skip_blank_lines=True)  # Ignore blank lines
        data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

        # Drop columns with all NaN values or irrelevant columns
        data = data.dropna(axis=1, how='all')  # Remove empty columns

        # If the dataset has extra spaces or inconsistent column names, we'll remove the extra spaces
        return data
    except pd.errors.ParserError as e:
        st.error(f"Error reading CSV: {e}")
        return None

data = load_data()

if data is not None:
    st.write("### Example Experimental Data")
    st.write(data.head())

    # Train a simple model using the cleaned data
    X = data[['Sand', 'Silt', 'Clay', 'Cement']]  # Adjust this based on your dataset
    y = data['Compressive strength']  # Replace with correct target column name if needed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make prediction for user input
    prediction = model.predict(input_data[['Sand', 'Silt', 'Clay', 'Cement']])[0]
    st.write(f"### Predicted Compressive Strength: {prediction:.2f} MPa")

    # Visualization: Predicted vs Actual
    st.write("### Model Performance")
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel("Actual Strength (MPa)")
    ax.set_ylabel("Predicted Strength (MPa)")
    st.pyplot(fig)

# Future Enhancement: Upload new experimental data
st.write("Upload new experimental data below to refine the model.")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data)
    
    # Optionally, retrain the model with the new data
    if 'Compressive strength' in new_data.columns:
        X_new = new_data[['Sand', 'Silt', 'Clay', 'Cement']]
        y_new = new_data['Compressive strength']
        
        # Train new model with the uploaded data
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
        model_new = RandomForestRegressor(random_state=42)
        model_new.fit(X_train_new, y_train_new)
        
        # Make new prediction based on updated model
        prediction_new = model_new.predict(input_data[['Sand', 'Silt', 'Clay', 'Cement']])[0]
        st.write(f"### New Predicted Compressive Strength with Uploaded Data: {prediction_new:.2f} MPa")
        
        # Visualize the new model's performance
        y_pred_new = model_new.predict(X_test_new)
        fig_new, ax_new = plt.subplots()
        sns.scatterplot(x=y_test_new, y=y_pred_new, ax=ax_new)
        ax_new.plot([y_test_new.min(), y_test_new.max()], [y_test_new.min(), y_test_new.max()], 'r--', linewidth=2)
        ax_new.set_xlabel("Actual Strength (MPa)")
        ax_new.set_ylabel("Predicted Strength (MPa)")
        st.pyplot(fig_new)
