import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# ------------------------------
# Model Definition
# ------------------------------

class FoodCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(FoodCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# ------------------------------
# Load Model
# ------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists("class_names.txt"):
        st.error("class_names.txt not found. Please run the training notebook first.")
        return None, None
        
    class_names = sorted([
        folder for folder in open("class_names.txt").read().splitlines()
    ])
    num_classes = len(class_names)

    model = FoodCNN(num_classes=num_classes)
    
    if os.path.exists("food_model.pth"):
        model.load_state_dict(torch.load("food_model.pth", map_location="cpu", weights_only=True))
    else:
        st.error("food_model.pth not found. Please run the training notebook first.")
        return None, class_names

    model.eval()
    return model, class_names

model, class_names = load_model()


# ------------------------------
# Transform (EfficientNet Standard)
# ------------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ------------------------------
# Calorie Database
# ------------------------------

calorie_dict = {
    "Bread": 265,
    "Dairy product": 100,
    "Dessert": 350,
    "Egg": 155,
    "Fried food": 300,
    "Meat": 250,
    "Noodles-Pasta": 131,
    "Rice": 130,
    "Seafood": 100,
    "Soup": 50,
    "Vegetable-Fruit": 60
}

# ------------------------------
# UI
# ------------------------------

st.title("🍔 Food Image Classification & Calorie Estimator (SOTA)")
st.write("Upload a food image to predict the dish and estimate calories.")

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    calories = calorie_dict.get(predicted_class, "Not Available")

    st.subheader("Prediction Result")
    st.write(f"🍽 Food: **{predicted_class}**")
    st.write(f"📊 Confidence: **{confidence_score:.2f}%**")
    st.write(f"🔥 Estimated Calories (per 100g): **{calories} kcal**")
