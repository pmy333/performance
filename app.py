import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model_pkl", "rb"))  # make sure filename matches

# Page config
st.set_page_config(page_title="ML Predictor", page_icon="🚀", layout="centered")

# Title
st.title("🚀 Smart ML Prediction App")
st.markdown("### Enter feature values below")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info("This app predicts outcomes using a trained ML model.")

# Detect number of features
try:
    n_features = model.n_features_in_
except:
    n_features = 5  # fallback

st.write(f"🔢 Model expects **{n_features} features**")

# Create dynamic inputs
inputs = []

cols = st.columns(2)

for i in range(n_features):
    with cols[i % 2]:
        val = st.number_input(f"Feature {i+1}", value=0.0)
        inputs.append(val)

# Convert input
input_data = np.array([inputs])

# Predict button
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit ❤️")
