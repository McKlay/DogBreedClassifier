# 🐶 Dog Breed Classifier V2

A lightweight image classifier built using **MobileNetV2** and transfer learning in PyTorch.

This model can classify images into 20 dog breeds with high accuracy.  
It is deployed using **Gradio** on Hugging Face Spaces.

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

## 🧠 Example Classes

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

This app is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/McKlay/DogBreedClassifier) using Gradio.  
It loads class names dynamically from `data/train/`, so folders must be present (can be empty).

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

Feel free to fork and modify this space!
