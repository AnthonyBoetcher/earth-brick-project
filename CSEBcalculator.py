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

st.write("### Input Ratios")
st.write(input_data)

# Example dataset (replace this with experimental data)
# Columns: 'Sand', 'Silt', 'Clay', 'Cement', 'Strength'
# Load data or use a placeholder
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'Sand': np.random.randint(30, 70, 100),
        'Silt': np.random.randint(10, 50, 100),
        'Clay': np.random.randint(10, 40, 100),
        'Cement': np.random.randint(0, 15, 100),
        'Strength': np.random.uniform(2, 10, 100)  # Strength in MPa
    })
    return data

data = load_data()
st.write("### Example Experimental Data")
st.write(data.head())

# Train a simple model
X = data[['Sand', 'Silt', 'Clay', 'Cement']]
y = data['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make prediction
prediction = model.predict(input_data)[0]
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

# Future Enhancement
st.write("Upload new experimental data below to refine the model.")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data)
    
    # Optionally, retrain the model with the new data
    if 'Strength' in new_data.columns:
        X_new = new_data[['Sand', 'Silt', 'Clay', 'Cement']]
        y_new = new_data['Strength']
        
        # Train new model
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
        model_new = RandomForestRegressor(random_state=42)
        model_new.fit(X_train_new, y_train_new)
        
        # Make new prediction based on updated model
        prediction_new = model_new.predict(input_data)[0]
        st.write(f"### New Predicted Compressive Strength with Uploaded Data: {prediction_new:.2f} MPa")
        
        # Visualize the new model's performance
        y_pred_new = model_new.predict(X_test_new)
        fig_new, ax_new = plt.subplots()
        sns.scatterplot(x=y_test_new, y=y_pred_new, ax=ax_new)
        ax_new.plot([y_test_new.min(), y_test_new.max()], [y_test_new.min(), y_test_new.max()], 'r--', linewidth=2)
        ax_new.set_xlabel("Actual Strength (MPa)")
        ax_new.set_ylabel("Predicted Strength (MPa)")
        st.pyplot(fig_new)
