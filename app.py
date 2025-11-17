import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="🍷 Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Model & Data ---

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained model pipeline."""
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please make sure it's in the correct folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

# Use st.cache_data to load and process data
@st.cache_data
def load_slider_data(data_path):
    """Loads the red wine dataset to get min/max values for sliders."""
    try:
        # Load the dataset
        df = pd.read_csv(data_path, sep=',')
        
        # This is a check in case the separator is wrong
        if len(df.columns) <= 1:
            df = pd.read_csv(data_path, sep=';')
            
        # Get min/max for the 11 original features
        df_sliders = df.drop(columns=['quality'], errors='ignore')
        return df_sliders.describe().loc[['min', 'max']]
        
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}. Using default ranges.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the data: {e}")
        return None

# Load the model and slider data
pipeline = load_model('wine_model_pipeline_V2.pkl')
slider_data = load_slider_data('winequality-red.csv')

# --- 3. Feature Engineering Function ---
# THIS IS THE MOST IMPORTANT PART
# It *must* match the engineering done in your Colab notebook for V2
def engineer_features(input_df):
    """Creates the two new features that the V2 model expects."""
    df = input_df.copy()
    
    # 1. Create 'total_acidity'
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
    
    # 2. Create 'acidity_ph_ratio'
    # Use 1e-6 to prevent division by zero, just in case
    df['acidity_ph_ratio'] = df['total_acidity'] / (df['pH'] + 1e-6)
    
    # Return the DataFrame with all 13 features
    # (11 original + 2 new)
    return df

# --- 4. Sidebar GUI for User Input ---
st.sidebar.header("Input Wine Features")
st.sidebar.markdown("Use the sliders to input the wine's properties.")

# Helper function to create sliders
def create_slider(column_name, slider_info):
    """Creates a single slider for a feature."""
    if slider_info is not None:
        min_val = float(slider_info[column_name]['min'])
        max_val = float(slider_info[column_name]['max'])
        default_val = float(slider_info[column_name].mean())
    else:
        # Fallback values if data loading failed
        min_val, max_val, default_val = 0.0, 100.0, 50.0
    
    return st.slider(
        label=column_name.replace("_", " ").title(),
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=0.01 # Add a small step for precision
    )

# Create sliders in two columns for a cleaner look
col1, col2 = st.sidebar.columns(2)
with col1:
    fixed_acidity = create_slider('fixed acidity', slider_data)
    volatile_acidity = create_slider('volatile acidity', slider_data)
    citric_acid = create_slider('citric acid', slider_data)
    residual_sugar = create_slider('residual sugar', slider_data)
    chlorides = create_slider('chlorides', slider_data)

with col2:
    free_sulfur_dioxide = create_slider('free sulfur dioxide', slider_data)
    total_sulfur_dioxide = create_slider('total sulfur dioxide', slider_data)
    density = create_slider('density', slider_data)
    pH = create_slider('pH', slider_data)
    sulphates = create_slider('sulphates', slider_data)

# Alcohol slider below the columns
alcohol = st.sidebar.slider(
    label='Alcohol',
    min_value=float(slider_data['alcohol']['min']) if slider_data is not None else 8.0,
    max_value=float(slider_data['alcohol']['max']) if slider_data is not None else 15.0,
    value=float(slider_data['alcohol'].mean()) if slider_data is not None else 10.0,
    step=0.01
)

# --- 5. Main Page: Display Input & Prediction ---
st.title("🍷 Wine Quality Prediction System")
st.markdown("This app predicts if a wine is **Good (1)** or **Bad (0)** quality.")

# Collect user inputs into a dictionary
input_data = {
    'fixed acidity': fixed_acidity,
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_sulfur_dioxide,
    'total sulfur dioxide': total_sulfur_dioxide,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

st.subheader("Your Input Features:")
st.dataframe(input_df)

# Prediction button
if st.sidebar.button("✨ Predict Quality"):
    if pipeline is not None:
        try:
            # --- This is the "try-catch" you wanted ---
            
            # 1. Apply feature engineering
            engineered_df = engineer_features(input_df)
            
            # 2. Make prediction
            prediction = pipeline.predict(engineered_df)
            probability = pipeline.predict_proba(engineered_df)
            
            pred_class = prediction[0]
            pred_prob = probability[0][pred_class]
            
            st.subheader("🎉 Prediction Result")
            
            # Display based on the binary prediction
            if pred_class == 1:
                st.success(f"**Prediction: GOOD Quality Wine**")
            else:
                st.error(f"**Prediction: BAD Quality Wine **")
                
            st.metric(label="Prediction Confidence", value=f"{pred_prob * 100:.2f}%")
            
            # Show probabilities for both classes
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame(probability, columns=['Bad (0)', 'Good (1)'], index=['Probability'])
            st.dataframe(prob_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("This is the 'try-except' block in action. Check feature engineering logic.")
    else:
        st.error("Model is not loaded. Cannot make a prediction.")

st.sidebar.markdown("---")
st.sidebar.write("App designed for the `wine_model_pipeline_V2.pkl` model.")