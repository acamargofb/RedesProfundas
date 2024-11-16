import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse

# Define AlexNet architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Function to classify an image
def classify_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained AlexNet model
    model = AlexNet()
    model.load_state_dict(torch.load('alexnet_cifar10.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet expects input size of 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)

    # Get predicted class
    _, predicted = torch.max(output, 1)

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"The image is classified as: {class_names[predicted.item()]}")

# Main entry point for command-line execution
if __name__ == '__main__':
# Ejemplo de uso: python AlexNetClassCIFAR10_v1.py dog1.jpg

    parser = argparse.ArgumentParser(description='Classify an image using a pre-trained AlexNet model on CIFAR-10.')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    args = parser.parse_args()

    classify_image(args.image_path)
