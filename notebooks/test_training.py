# 📦 Imports
import sys, os
from pathlib import Path

# Set root directory explicitly
ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))  # Add project root to sys.path

from src.utils import load_model, plot_training_curves, predict_image, load_image
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random

# ✅ Load trained model
model_path = ROOT / "models/mobilenet_v2.pth"
train_dir = ROOT / "data" / "train"
val_dir = ROOT / "data" / "val"

class_names = sorted(os.listdir(train_dir))
model = load_model(str(model_path), num_classes=len(class_names))

# ✅ Plot training curves
#plot_training_curves()

# 🖼️ Show sample prediction
chosen_class = random.choice(class_names)
chosen_img = random.choice(os.listdir(val_dir / chosen_class))
img_path = val_dir / chosen_class / chosen_img

image = Image.open(img_path).convert("RGB")
image_tensor = load_image(img_path)  # This handles resizing, tensor conversion, normalization
predicted_class, confidence = predict_image(model, image_tensor, class_names)


print(f"📷 Image: {chosen_img}")
print(f"✅ Predicted: {predicted_class} ({confidence:.2f})")

# 🖼️ Visualize
plt.imshow(image)
plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
plt.axis("off")
plt.show()
