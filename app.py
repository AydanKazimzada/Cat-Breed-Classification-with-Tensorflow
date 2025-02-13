import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import streamlit as st


st.header('Cat Breed Classification Model🐾')

cat_names = ['British Shorthair', 'Persian', 'Scottish Fold', 'Siamese', 'Sphynx']

model = load_model('cat_breed_model.keras')

def classify_images(image_path):   
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    prediction = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(prediction[0])

    outcome = f'It is a lovely {cat_names[np.argmax(result)]} cat with a score of {np.max(result) * 100:.2f}%.'
    return outcome

st.markdown(
    """
    <style>
        .upload-text {
            font-size: 24px !important; /* Enforce large text size */
            font-weight: bold;
            color: var(--text-color); /* Adapts to Streamlit theme */
            text-align: left; /* Align text to the left */
        }
        .result-text {
            font-size: 20px !important; /* Increase result text size */
            color: var(--text-color); /* Adapts to Streamlit theme */
            text-align: left; /* Align text to the left */
        }
        .cute-text {
            font-size: 26px !important;
            color: var(--text-color);
            text-align: left;
        }
    </style>
    <p class="upload-text">📤 Upload Cute Cat Image :)</p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:

    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True) 
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())


    st.image(uploaded_file, width=400)

    cute_phrase = "Purrfect choice! 😻"
    st.markdown(f'<p class="cute-text">{cute_phrase}</p>', unsafe_allow_html=True)

    result = classify_images(file_path)
    st.markdown(f'<p class="result-text">{result}</p>', unsafe_allow_html=True)
