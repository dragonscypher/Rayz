import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from transformers import BartTokenizer, BartForConditionalGeneration
import requests
from bs4 import BeautifulSoup

# ✅ Initialize Flask App
app = Flask(__name__, template_folder="templates")

# ✅ Define Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Define Model Path (EfficientNet-B0)
MODEL_PATH = "EfficientNet_B0_best_93.44.pth"

# ✅ Load Model Checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
num_classes = checkpoint["classifier.1.weight"].shape[0]  # Auto-detect class count

# ✅ Load EfficientNet-B0 Model
model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)  # Dynamically adjust
model.load_state_dict(checkpoint)
model.eval()

# ✅ Define Disease Classes (Ensuring "No Finding" Is Handled Properly)
classes = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
]

# ✅ Load BART Model for Report Generation
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Function: Preprocess Image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# ✅ Function: Predict Disease Probabilities
def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities.numpy()

# ✅ Function: Visualize Predictions
def visualize_predictions(image_path, probabilities):
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = np.array(probabilities)[sorted_indices]
    sorted_classes = np.array(classes)[sorted_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_classes, y=sorted_probs, palette="coolwarm")
    plt.xlabel("Disease Type")
    plt.ylabel("Prediction Probability")
    plt.title("Disease Probability Distribution")
    plt.xticks(rotation=45, ha="right")

    plot_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# ✅ Function: Retrieve a Health Info Link from Google
# ✅ Function: Generate Google Search Link for Conditions
def get_health_info_link(condition):
    base_url = "https://www.google.com/search?q="
    query = f"{condition} chest X-ray explanation"
    search_url = base_url + "+".join(query.split())  # Convert spaces to '+'
    return search_url  # Return direct Google search link




# ✅ Function: Generate Structured Report using BART
def generate_structured_report(predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    top_diseases = [classes[i] for i in sorted_indices[:3]]

    # ✅ Handling "No Finding" Case
    no_finding_index = classes.index("No Finding")
    no_finding_prob = predictions[no_finding_index]
    other_max_prob = max([predictions[i] for i in sorted_indices if i != no_finding_index])

    if no_finding_prob > 0.9 and other_max_prob < 0.2:  # ✅ 90% confidence threshold
        return "✅ The chest X-ray appears normal. No significant abnormalities detected.", {}

    # ✅ Get Google links for additional information
    health_links = {disease: get_health_info_link(disease) for disease in top_diseases}

    # ✅ Construct structured text
    structured_text = f"""
    The patient has the following conditions based on the chest X-ray analysis: {', '.join(top_diseases)}.
    
    Findings:
    - Lungs show signs of {top_diseases[0]}, which suggests possible complications.
    - Mild to moderate {top_diseases[1]} is observed in the affected area.
    - Clinical assessment is required for {top_diseases[2]}, as it may require further imaging.
    
    Impression:
    - The findings are consistent with the listed conditions.
    - Recommend follow-up with a radiologist for further evaluation.

    🔍 To learn more about these conditions:
    - {top_diseases[0]}: [Read here]({health_links[top_diseases[0]]})
    - {top_diseases[1]}: [Read here]({health_links[top_diseases[1]]})
    - {top_diseases[2]}: [Read here]({health_links[top_diseases[2]]})
    """

    # ✅ Summarize with BART
    input_ids = bart_tokenizer.encode(structured_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(input_ids, max_length=150, min_length=50, do_sample=True, temperature=0.7)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary, health_links  # ✅ RETURN health_links!

# ✅ Flask Routes
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # ✅ Process Image
            image_tensor = preprocess_image(file_path)
            probabilities = predict(image_tensor)
            plot_path = visualize_predictions(file_path, probabilities)

            # ✅ Generate Report
            structured_report, health_links = generate_structured_report(probabilities)

            # ✅ Generate Google Search Links
            health_links = {disease: get_health_info_link(disease) for disease in health_links.keys()}

            return render_template("result.html", image_path=file_path, plot_path=plot_path, report=structured_report, health_links=health_links)

    return render_template("index.html")



# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
