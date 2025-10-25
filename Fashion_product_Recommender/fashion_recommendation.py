import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from PIL import Image
import pickle
from sklearn.neighbors import NearestNeighbors
import json
import pandas as pd

import os, pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  
emb_path = BASE_DIR / "embiddings.pkl"
fn_path  = BASE_DIR / "filenames.pkl"
images_csv = BASE_DIR / "images.csv"


features_list = np.array(pickle.load(open(emb_path,'rb')))
filenames = pickle.load(open(fn_path,'rb'))
mapping_df = pd.read_csv(images_csv)
mapping_dict = dict(zip(mapping_df["filename"], mapping_df["link"]))


def fetching_json(index):
    with open(r"C:\Users\divya\Desktop\RecSys\RecSys\Fashion-product-recommender-main\data\fashion-dataset\styles\{}.json".format(index)) as f:
        return json.load(f)

def save_file(file):
    try:
        upload_folder = "uploaded"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path   
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Model setup
model = ResNet50(include_top = False, weights='imagenet', input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

def extract_function(model, img):
    imag = img.convert("RGB").resize((224, 224))
    imag_array = image.img_to_array(imag)
    imag_expand = np.expand_dims(imag_array,axis=0)
    processed_image = preprocess_input(imag_expand)
    result = model.predict(processed_image).flatten()
    return result / norm(result)

def recommend(features_list, feature):
    neighbor = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
    neighbor.fit(features_list)
    distance, index = neighbor.kneighbors([feature])
    return index

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fashion Recommender", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center; color:#ff4b4b;"> Fashion Product Recommender</h1>
    <p style="text-align:center; font-size:18px; color:gray;">
    Upload an image and get similar fashion recommendations instantly.
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Upload Image to find similar products")
file = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg','png'])

if file is not None:
    st.write("### Successfully Uploaded Image")
    #saved_path = save_file(file)

    
    img = Image.open(file)
    st.image(img, width=350, caption="Uploaded Image")

    with st.spinner("Extracting features..."):
        feature = extract_function(model, img)
    st.success("Features extracted!")

    with st.spinner("Finding similar products..."):
        index = recommend(features_list, feature)
    st.success("Recommendations ready!")

    st.markdown("##Top Recommendations")
    cols = st.columns(5)

    for i, col in enumerate(cols):
        idx = index[0][i]
        file_path = filenames[idx]
        file_name = file_path.split("\\")[-1]
        #st.write(file_name)
        
        mapped_link = mapping_dict.get(file_name)
        
        

        product_id = os.path.splitext(os.path.basename(file_path))[0]

        try:
            data_json = fetching_json(product_id)
            caption = data_json['data'].get('productDisplayName', f"Item {i+1}")
        except:
            caption = f"Recommendation {i+1}"
        
        #st.write(mapped_link)
        if mapped_link:  # Only show if link exists
            with col:
                st.image(mapped_link, caption=caption)
        else:
            with col:
                st.write("Image not found")

       

    st.success("Done! Explore the recommendations above.")
