import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# ---------------------------
# Load the model
# ---------------------------
try:
    model = load_model("cat_dog_cnn.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

def predict(image):

    img=image.convert("RGB")

    #resize the image
    img=img.resize((128,128))

    #noramlize
    img=np.array(img)/255.0

    #ewshape for the model impuy type
    img=img.reshape(1,128,128,3)

    #predict
    result=model.predict(img)
    pre_result=np.argmax(result,axis=1)[0]  # cat=0,dog=1
    return pre_result


st.title("cat dog predictor")
st.write("upload image")

# File uploader
file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if file is not None:
    image = Image.open(file)#open the image
    st.image(image, caption="Uploaded Image", width=200)

    # Prediction button
    if st.button("Predict Now"):
        prediction = predict(image)

        if prediction == 0:
            st.success("Predicted Animal: CAT")
        else:
            st.success("Predicted Animal: DOG")
