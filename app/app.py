# app/app.py

import os
import gradio as gr
import torch
from PIL import Image

from src.model import get_model
from src.utils import load_image, predict_image

# ---- Paths and Configs ----
MODEL_PATH = "models/mobilenet_v2.pth"
DATA_PATH = "data/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Dynamically Load Class Names from Folder ----
CLASS_NAMES = sorted(os.listdir(DATA_PATH))

# ---- Load Model ----
model = get_model(num_classes=len(CLASS_NAMES), model_name="mobilenet_v2", pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---- Prediction Function ----
def classify_dog(image):
    tensor = load_image(image)
    label, confidence = predict_image(model, tensor, CLASS_NAMES, DEVICE)
    return f"{label} ({confidence:.2%})"

# ---- Gradio Interface ----
title = "🐶 Dog Breed Classifier V2"
description = "Upload a dog image to classify its breed using a fine-tuned MobileNetV2 model."

demo = gr.Interface(
    fn=classify_dog,
    inputs=gr.Image(type="pil", label="Upload a dog image"),
    outputs=gr.Text(label="Predicted Breed"),
    title=title,
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
