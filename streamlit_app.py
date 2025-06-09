import os
import json
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import subprocess

# Додатковий імпорт, щоб мати змогу викликати train.py напряму
import sys

st.set_page_config(page_title="Aircraft Recognition", layout="centered")

mode = st.sidebar.radio("Mode", ["Inference", "Train"])

if mode == "Inference":
    st.title("Aircraft Recognition System")
    st.write("Upload images and select a trained model to classify them.")

    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    selected_model = st.selectbox("Choose model", model_files)

    class_file = os.path.join(model_dir, "class_names.json")
    class_names = json.load(open(class_file)) if os.path.exists(class_file) else None

    uploaded_files = st.file_uploader(
        "Upload image(s)", type=["png","jpg","jpeg"], accept_multiple_files=True
    )

    if st.button("Recognize"):
        if not uploaded_files:
            st.warning("Please upload at least one image.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
            model.load_state_dict(torch.load(os.path.join(model_dir, selected_model), map_location=device))
            model.to(device).eval()

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                st.subheader(f"Image {i+1}: {f.name}")
                st.image(img, use_column_width=True)
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor)[0], dim=0).cpu().numpy()
                df = pd.DataFrame({"Class": class_names, "Probability": (probs*100).round(2)})
                df = df.sort_values("Probability", ascending=False)
                st.write("### Probabilities (%)")
                st.table(df)
                st.bar_chart(df.set_index("Class")["Probability"])

elif mode == "Train":
    st.title("Train a New Model")
    st.write("Configure training parameters and launch training right here.")

    data_dir = st.text_input("Data directory", value="data")
    output_dir = st.text_input("Output (models) directory", value="models")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32)
    lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
    save_epochs = st.text_input("Save epochs (space-separated)", value="10")

    if st.button("Start Training"):
        save_list = " ".join(save_epochs.split())
        cmd = [
            sys.executable, "train.py",
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--save_epochs", *save_list.split()
        ]
        st.write("Running command:", " ".join(cmd))
        # Запускаємо окремий процес, щоб не блокувати Streamlit
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            st.text(line)
        proc.wait()
        st.success("Training completed!")
