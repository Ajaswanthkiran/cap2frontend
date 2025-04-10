import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Parkinson's Detection", page_icon="🧠", layout="wide")


precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

def f1_score_metric(y_true, y_pred):
    # Update the state of the metrics with the current batch
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)

    # Get the current values of precision and recall
    precision_value = precision.result()
    recall_value = recall.result()

    # Calculate the F1 score
    f1_score = 2 * ((precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon()))

    # Reset the state of the metrics for the next batch
    precision.reset_state()  # Use reset_states instead of reset_state
    recall.reset_state()     # Use reset_states instead of reset_state

    return f1_score

# Load the trained model
@st.cache_resource
def load_trained_model():
    try:
       # Now load the model with the custom function
        model = load_model(r'C:\Users\16307\Desktop\Cap2\hrl_hi_model1.keras',custom_objects={'f1_score_metric': f1_score_metric})

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Validate if the uploaded image is a proper brain MRI
def validate_mri_image(img):
    # Check if the image is grayscale (black and white)
    if len(np.array(img).shape) > 2:
        st.error("Invalid! Please upload an MRI brain image.")
        return False  # Stop further checks

    # Convert to grayscale (if not already)
    img_gray = img.convert('L')
    img_array = np.array(img_gray)

    # Check grayscale variance (MRI scans typically have higher variance)
    grayscale_variance = np.var(img_array)
    if grayscale_variance < 100:  # Adjust threshold as needed
        st.error("Invalid! Please upload an MRI brain image.")
        return False  # Stop further checks

    return True  # If all checks pass, it's a valid MRI brain image

# Preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize image
    img = img.convert("RGB")  # Ensure it has 3 channels

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Function to predict Parkinson’s disease
def predict_disease(model, img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    result = "🧠 Parkinson Detected" if prediction[0][0] < 0.5 else "✅ No Parkinson Detected"
    return result  # Removed confidence from the return

# Custom CSS Styling
st.markdown("""
    <style>
        .main {background-color: #f0f5f9;}
        .stButton>button {width: 100%; border-radius: 6px; font-size: 18px; padding: 10px;}
        .header {text-align: center; font-size: 28px; font-weight: bold; color: #004080;}
        .subheader {text-align: center; font-size: 20px; font-weight: bold; color: #0080ff;}
    </style>
""", unsafe_allow_html=True)

def main():
    model = load_trained_model()
    print("Model:", model)

    # Sidebar with App Information
    st.sidebar.title("🩺 Parkinson's Detection")
    st.sidebar.markdown("""
        This tool helps detect **Parkinson's Disease** based on MRI scans.
        
        **How it works:**
        - Upload an MRI scan (JPG, JPEG, PNG)
        - Click **Predict** to get a diagnosis
        - Read information about Parkinson's below
    """)
    st.sidebar.markdown("""
        **Disclaimer:**
        This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.
    """)

    # Main Content Layout
    col1, col2 = st.columns([1, 2])

    # Left Panel - Upload Image & Prediction
    with col1:
        st.title("Upload Section")
        st.markdown("Please upload a brain MRI image in `.jpg`, `.jpeg`, or `.png` format.")
        uploaded_file = st.file_uploader("Choose an MRI Scan", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            img = Image.open(uploaded_file)  
            st.image(img, caption="Uploaded Image", use_container_width=True)

            # Validate the MRI image
            if not validate_mri_image(img):  # Stop if validation fails
                return  # Exit the function immediately

            # If the image is valid, proceed with prediction
            if st.button("Predict", key="predict_btn"):
                with st.spinner("Analyzing the MRI scan..."):
                    result = predict_disease(model, img)  # Removed confidence
                st.success(f"Prediction: {result}")

                # Feedback Mechanism
                

    # Right Panel - Information Section
    with col2:
        st.markdown('<div class="header">🧠 About Parkinson’s Disease</div>', unsafe_allow_html=True)
        st.write("""
                Parkinson's disease is a progressive neurodegenerative disorder that primarily affects movement due to the loss of dopamine-producing neurons in the brain. Symptoms include tremors, stiffness, slowness of movement, and balance issues, along with non-motor symptoms like cognitive decline, mood disorders, and sleep disturbances. While not directly fatal, complications such as falls, pneumonia, or swallowing difficulties can be life-threatening. Globally, Parkinson's affects about 1% of people aged 60 and above, with approximately 8.5 million cases worldwide. Men are 1.5 times more likely to develop the disease than women, and the number of cases is expected to double by 2040 due to aging populations. Risk factors include age, genetics, environmental exposures, and lifestyle choices. Although there is no cure, treatments like medications, therapies, and lifestyle changes can help manage symptoms and improve quality of life. Raising awareness and advancing research are essential to finding better treatments and ultimately a cure.
            """)
        with st.expander("🔍 Learn More About Parkinson’s Disease"):
           

            # Causes
            st.markdown('<div class="subheader">🔍 Causes of Parkinson’s</div>', unsafe_allow_html=True)
            st.write("""
            - � **Genetics**: Certain inherited gene mutations can increase the risk.
            - 🌍 **Environmental Factors**: Pesticides, toxins, and heavy metals exposure.
            - 🎂 **Age Factor**: More common in individuals over 60.
            - ⚡ **Dopamine Loss**: Affects brain function and movement.
            """)

            # Symptoms
            st.markdown('<div class="subheader">⚠️ Recognizing Symptoms</div>', unsafe_allow_html=True)
            st.write("""
            - 🤲 **Tremors**: Involuntary shaking of hands, fingers, or legs.
            - 🚶 **Slow Movement (Bradykinesia)**: Reduced ability to move quickly.
            - 💪 **Muscle Rigidity**: Stiffness causing movement difficulty.
            - 🗣️ **Speech Issues**: Soft, slurred, or monotone speech.
            """)
            # Diagnosis
            st.markdown('<div class="subheader">🩺 How is Parkinson’s Diagnosed?</div>', unsafe_allow_html=True)
            st.write("""
            - 👨‍⚕️ **Neurological Tests** to assess motor functions.
            - 🧠 **MRI or CT Scans** to examine brain abnormalities.
            - 🏥 **DaT Scan** for evaluating dopamine levels.
            """)
             # Treatments
            st.markdown('<div class="subheader">💊 Treatment & Management</div>', unsafe_allow_html=True)
            st.write("""
            - 💊 **Medications** to balance dopamine.
            - 🏃 **Physical Therapy** to improve movement.
            - 🍎 **Healthy Diet** rich in antioxidants and omega-3s.
            """)

            # Lifestyle
            st.markdown('<div class="subheader">🍏 Healthy Living Tips</div>', unsafe_allow_html=True)
            st.write("""
            - 🏋️ **Exercise Regularly** (Yoga, walking, or stretching).
            - 🍽️ **Maintain a Nutritious Diet** for brain health.
            - ⏳ **Follow Medical Advice** & routine check-ups.
            """)

       

if __name__ == "__main__":
    
    main()