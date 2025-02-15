import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import multiprocessing

# CBAM Attention Module
class CBAM(torch.nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels // reduction, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels // reduction, channels, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * (avg_out + max_out)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class EnhancedEfficientNet(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(EnhancedEfficientNet, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cbam = CBAM(channels=1280)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model.features(x)
        x = self.cbam(x) * x
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

if __name__ == "__main__":
    multiprocessing.freeze_support()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnhancedEfficientNet(num_classes=15)

 
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    data_dir = "normalized_images_final1"
    dataset = ImageFolder(root=data_dir, transform=transform)

  
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/5], Loss: {total_loss:.4f}")


    torch.save(model.state_dict(), "EfficientNet_B0_CBAM.pth")
    print("Fine-tuned model saved as EfficientNet_B0_CBAM.pth")

    model.load_state_dict(torch.load("EfficientNet_B0_CBAM.pth", map_location=device))
    model.eval()

    # Visualization
    def grad_cam_visualization(image_path):
        target_layer = model.module.model.features[-1] if isinstance(model, torch.nn.DataParallel) else model.model.features[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        input_tensor = preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        grayscale_cam = cam(input_tensor=input_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam[0]), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

        plt.figure(figsize=(8, 6))
        plt.imshow(superimposed_img)
        plt.axis("off")
        plt.title("Grad-CAM: Model Attention Map")
        plt.show()


    grad_cam_visualization("nhih/images_003/images/00003924_000.png")
