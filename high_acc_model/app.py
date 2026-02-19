import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

st.set_page_config(page_title="High Accuracy Food Classifier", page_icon="🍔")

# ------------------------------
# Model Definition (Must Match train.py)
# ------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists("class_names.txt"):
        st.error("class_names.txt not found. Please run the training script first.")
        return None, None
        
    class_names = sorted([
        name.strip() for name in open("class_names.txt").readlines()
    ])
    num_classes = len(class_names)

    # Reconstruct the model
    # We don't need weights=IMG... here because we load our own, but it doesn't hurt to initialize similarly
    # efficiently.
    model = models.efficientnet_b0(weights=None) 
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2), 
        nn.Linear(512, num_classes)
    )
    
    map_location = torch.device('cpu')
    model_path = "effective_food_model.pth"
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=map_location)
            model.load_state_dict(state_dict)
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
            return None, class_names
    else:
        st.error(f"{model_path} not found. Please run the training script first.")
        return None, class_names

    model.eval()
    return model, class_names

model, class_names = load_model()


# ------------------------------
# Transform (Validation/Inference)
# ------------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------
# Calorie Database (Legacy)
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

# Moved to top


st.title("🍔 High Accuracy Food Classifier (EfficientNet)")
st.write("Upload a food image to predict the dish with high confidence.")

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
    calories = calorie_dict.get(predicted_class, "N/A")

    st.subheader("Prediction Result")
    st.success(f"🍽 Food: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence_score:.2f}%**")
    st.warning(f"🔥 Estimated Calories (per 100g): **{calories} kcal**")
