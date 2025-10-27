
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_fish_classification_model.h5')
    return model

# Load class names (update these with your actual class names)
class_names = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Resize image to match model input
    image = image.resize((224, 224))
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def plot_predictions(predictions, class_names):
    """Create a bar chart of prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(class_names))

    # Sort predictions for better visualization
    sorted_indices = np.argsort(predictions[0])[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = predictions[0][sorted_indices]

    bars = ax.barh(y_pos, sorted_probs, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_classes)
    ax.invert_yaxis()
    ax.set_xlabel('Confidence Score')
    ax.set_title('Prediction Confidence for All Fish Species')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2%}', ha='left', va='center')

    plt.tight_layout()
    return fig

# Main application
def main():
    # Header
    st.title("🐟 Fish Species Classification")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses deep learning to classify fish species from images. "
        "Upload a clear image of a fish, and the model will predict its species."
    )

    st.sidebar.markdown("### How to use:")
    st.sidebar.markdown("1. Upload a fish image (JPG, PNG, JPEG)")
    st.sidebar.markdown("2. Wait for the model to process")
    st.sidebar.markdown("3. View the prediction and confidence scores")

    st.sidebar.markdown("### Supported Species:")
    for i, species in enumerate(class_names, 1):
        st.sidebar.markdown(f"{i}. {species}")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Fish Image")
        uploaded_file = st.file_uploader(
            "Choose a fish image...", 
            type=["jpg", "png", "jpeg"],
            help="Upload a clear image of a fish for classification"
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess and predict
            with st.spinner('🔍 Analyzing the fish...'):
                # Preprocess image
                processed_image = preprocess_image(image)

                # Load model and predict
                model = load_model()
                predictions = model.predict(processed_image, verbose=0)

                # Get results
                confidence = np.max(predictions)
                predicted_class = class_names[np.argmax(predictions)]

    with col2:
        if uploaded_file is not None:
            st.subheader("Prediction Results")

            # Display main prediction
            if confidence > 0.7:
                st.success(f"**Prediction:** {predicted_class}")
            elif confidence > 0.4:
                st.warning(f"**Prediction:** {predicted_class}")
            else:
                st.error(f"**Prediction:** {predicted_class}")

            st.info(f"**Confidence:** {confidence:.2%}")

            # Confidence score interpretation
            if confidence > 0.9:
                st.success("🎯 High confidence - Very reliable prediction!")
            elif confidence > 0.7:
                st.info("👍 Good confidence - Reliable prediction")
            elif confidence > 0.5:
                st.warning("⚠️ Moderate confidence - Consider verifying")
            else:
                st.error("❌ Low confidence - Image might be unclear or not a fish")

            # Show all predictions
            st.subheader("All Predictions")
            fig = plot_predictions(predictions, class_names)
            st.pyplot(fig)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Built with TensorFlow & Streamlit** | "
        "**Multiclass Fish Image Classification Project**"
    )

if __name__ == "__main__":
    main()
