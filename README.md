---
title: DogBreedClassfier-mobilenetV2
emoji: 🐶
colorFrom: purple
colorTo: pink
sdk: gradio
app_file: app.py
pinned: true
---
[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?logo=huggingface&style=flat-square)](https://huggingface.co/spaces/McKlay/DogBreedClassfier-mobilenetV2)
[![Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange?logo=gradio&style=flat-square)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GitHub last commit](https://img.shields.io/github/last-commit/McKlay/DogBreedClassifier)
![GitHub Repo stars](https://img.shields.io/github/stars/McKlay/DogBreedClassifier?style=social)
![GitHub forks](https://img.shields.io/github/forks/McKlay/DogBreedClassifier?style=social)
![MIT License](https://img.shields.io/github/license/McKlay/DogBreedClassifier)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=McKlay.DogBreedClassifier)

# 🐶 DogBreedClassfier-mobilenetV2

A lightweight image classifier built using **MobileNetV2** and transfer learning in PyTorch.

This model can classify images into 20 dog breeds with high accuracy.  
It is deployed using **Gradio** on Hugging Face Spaces.

Stanford Dogs Dataset Link:
👉 [http://vision.stanford.edu/aditya86/ImageNetDogs/](http://vision.stanford.edu/aditya86/ImageNetDogs/)

---

## 📸 How to Use

1. Upload or drag a photo of a dog.
2. The model will predict its breed and show the confidence score.

---

## Model Details

- **Model**: MobileNetV2 (fine-tuned)
- **Validation Accuracy**: 93.35%
- **Classes**: 20 dog breeds
- **Framework**: PyTorch + Gradio

---

## Example Classes

- beagle
- papillon
- Weimaraner
- Staffordshire_bullterrier
- Chihuahua
- ... and more!

---

## 🗂️ Project Structure

7_DogBreedClassifierV2/  
├── app/  
│ └── app.py  
│  
├── data/  
│ ├── train/ ← 20 dog breed folders (used for label order)  
│ └── val/ ← validation set (not required for inference)  
│  
├── models/  
│ ├── mobilenet_v2.pth ← trained weights  
│ ├── efficientnet_b0.pth  
│ └── best_model.pth  
│  
├── src/  
│ ├── data_loader.py  
│ ├── model.py  
│ ├── train.py  
│ └── utils.py  
│  
├── requirements.txt  
└── README.md  


---

## 💻 Deployment

This app is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/McKlay/DogBreedClassfier-mobilenetV2) using Gradio.  
It loads class names dynamically from `data/train/`, so folders must be present (can be empty).

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

🤝 Credits  
Built by [Clay Mark Sarte](https://www.linkedin.com/in/clay-mark-sarte-283855147/) for deployment on Hugging Face.

Feel free to fork and modify this space!

---

Notes:

⚠️ This app uses NumPy 1.24.x to maintain compatibility with PyTorch and TorchVision C extensions. NumPy 2.x is not yet fully supported by these libraries.
