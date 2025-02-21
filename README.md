# 🌟 RayZ: AI-Powered Chest X-ray Analysis with RAG & EfficientNet_B0 🚀

![EfficientNet_B0](https://img.shields.io/badge/model-EfficientNet_B0-blue) ![Hugging Face](https://img.shields.io/badge/huggingface-model-orange) ![RAG](https://img.shields.io/badge/technology-RAG-green) ![Flask](https://img.shields.io/badge/backend-Flask-red)

## 🩺 Overview

**RayZ** is a **state-of-the-art AI-powered** **chest X-ray analysis tool** built with **EfficientNet_B0 & RAG (Retrieval-Augmented Generation)**. It can **detect 14 lung conditions** and generate a **structured radiology report** while **providing real-time medical search results**.

### 🧩 Why This Project?

The journey began when I **received a false-positive tuberculosis (TB) report** and had to wait for **delayed X-ray results** due to a holiday. Since I **couldn’t interpret the X-rays myself**, I decided to build an **AI-driven assistant** that could **automate X-ray analysis** and **bridge the gap between patients & radiologists**.

Now, **RayZ** offers an **end-to-end AI-driven radiology assistant** 🏥.

---

## 🔗 Model & Dataset Links

- **Hugging Face Model**: [RayZ EfficientNet_B0](https://huggingface.co/Dragonscypher/rayz_EfficientNet_B0)
- **Datasets Used**:
  - **NIH Chest X-ray Dataset** [📊](https://www.kaggle.com/datasets/nih-chest-xrays/data)
  - **NLMCXR Dataset** [📚](https://huggingface.co/datasets/Fakhraddin/NLMCXR)

---

## 🚀 Features & Capabilities

✅ **Detects 14 Lung Conditions**:  
Pneumonia, Atelectasis, Effusion, Infiltration, Mass, Nodule, Pneumothorax, etc.

✅ **Retrieval-Augmented Generation (RAG)** for generating **structured radiology reports**  

✅ **Real-time Medical Search Links** from **trusted sources**  

✅ **Uses EfficientNet_B0** for **high accuracy & efficiency**  

✅ **Trained on NIH & NLMCXR datasets** for robust predictions  

✅ **Flask Web App for user-friendly X-ray analysis**  

---

## 🛠️ Tech Stack & Skills Used

### **🤖 AI & Deep Learning**
- **EfficientNet_B0** (Image Classification)
- **Hugging Face Transformers** (BART for RAG-based structured reporting)
- **PyTorch** for model training & inference
- **CNN, DenseNet, ViT** (tested & compared)

### **⚡ Web Development**
- **Flask** (Python-based backend)
- **HTML, CSS, Jinja2** (Frontend)
- **Bootstrap & Dark UI Theme**

### **📊 Data Processing & Visualization**
- **Pandas, NumPy** (Data handling)
- **PIL (Pillow)** (Image processing)
- **Matplotlib, Seaborn** (Data Visualization)

### **🔍 Retrieval-Augmented Generation (RAG)**
- **BART-based NLP Model**
- **Google Search API for trusted health links**
- **Dynamic report summarization with BART**

---

## 📊 Why EfficientNet_B0?

After testing multiple models (**DenseNet121, ViT, CNNs**), **EfficientNet_B0_best_93.44** performed **better than others** in terms of:

| Model              | AUROC Score (Avg) |
|--------------------|------------------|
| **EfficientNet_B0** | **0.72 - 0.93** |
| DenseNet121       | 0.55 - 0.95      |
| ViT_Base          | 0.32 - 0.65      |

EfficientNet_B0 provided the **best balance of accuracy & speed** ⚡, making it ideal for **fast medical inference**.

---

## 🚀 How to Run the Project

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
2️⃣ Run the Flask Web App
```bash
python UI.py
```
3️⃣ Open in Browser

    Navigate to http://127.0.0.1:5000 🌐
    Upload a chest X-ray image
    Get predicted conditions + searchable medical info links

📖 How to Use the Model (Standalone)
```python
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModel
# Load model
model = AutoModel.from_pretrained("Dragonscypher/rayz_EfficientNet_B0")
model.eval()
# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Load image
image = Image.open("example_xray.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)
# Predict
with torch.no_grad():
    output = model(input_tensor)

print("Predicted probabilities:", output)
```
📊 Performance & Evaluation

    Training Data: NIH Chest X-ray & NLMCXR
    Loss Function: Cross-Entropy Loss
    Optimizer: Adam with Learning Rate Scheduling
    Batch Size: 32
    Epochs: 20
    Hardware Used: NVIDIA V100 GPU (AWS Cloud)
    AUROC Score: 0.72 - 0.93

⚠️ Limitations & Ethical Considerations

🚑 Not a diagnostic tool – AI does NOT replace radiologists.
🏥 Works best on frontal X-rays – lateral X-rays may not be well-supported.
🌍 May not generalize well – trained on US-based datasets, requiring fine-tuning for other populations.

Always consult a medical professional for accurate diagnosis!
📌 Future Enhancements

    📱 Mobile Deployment (ONNX / TensorFlow Lite)
    🌐 Cloud API for hospitals
    🧠 Multimodal AI (Combining X-ray + Clinical Notes)

🤝 Contributing

Want to improve RayZ? Feel free to submit a PR or raise an issue!
📜 License

This project is licensed under the MIT License.
⭐ Support & Citation

If you find this project useful, please consider giving it a ⭐ on GitHub!

