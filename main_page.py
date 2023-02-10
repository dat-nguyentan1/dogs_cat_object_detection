import streamlit as st 
import torch
from PIL import Image
import time
from pathlib import Path
import os 
from glob import glob

st.title('Dogs vs Cats Object Detection')

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

@st.cache(suppress_st_warning=True)
def load_model(weight_path=None):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path) 

imgs = []
weight_path = r'best.pt'

warn = st.warning("Loading model")

model = load_model(weight_path=weight_path)

if model:
    st.success('Load trained model succesfully', icon="✅")
    warn.empty()

uploaded_files = st.file_uploader("Choose image(s)", type=['png', 'jpg'], accept_multiple_files=True, help = "Support multiple images upload")

# for upload_file in uploaded_files:
if st.button("Predict"):
    if uploaded_files:
        for upload_file in uploaded_files:
            img = Image.open(upload_file)
            imgs.append(img)

        results = model(imgs, size=640)
        st.warning("Predicting")
        path = increment_path('output/exp')\

        # save predicted output
        results.save(save_dir = str(path))

        img_list = glob(os.path.join(str(path), "*.jpg"))

        for filename in img_list:
            print(filename)
            st.text(f"File names: {filename}")

            st.image(filename)

    else:
        st.error("No input files")