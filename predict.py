
import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

def get_input_args():
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict image class using a trained deep learning model.")
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--arch', type=str, choices=['resnet50', 'vgg16'], required=True, help='Model architecture')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath, arch):
    """Load model from checkpoint file dynamically based on architecture."""
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    if arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, len(checkpoint['class_to_idx'])),
            torch.nn.LogSoftmax(dim=1)
        )
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, len(checkpoint['class_to_idx'])),
            torch.nn.LogSoftmax(dim=1)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """Preprocess input image for the model."""
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image_path, model, top_k, device):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.to(device)
    model.eval()
    image = process_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities, classes = torch.exp(output).topk(top_k, dim=1)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_labels = [idx_to_class[i.item()] for i in classes[0]]
    
    return probabilities[0].tolist(), class_labels

def main():
    """Main function for image classification prediction."""
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from {args.checkpoint} using {args.arch} model...")
    model = load_checkpoint(args.checkpoint, args.arch)
    model.to(device)

    probabilities, classes = predict(args.image_path, model, args.top_k, device)
    
    if args.category_names and os.path.exists(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(cls, cls) for cls in classes]

    print("\nPrediction Results:")
    for i in range(len(classes)):
        print(f"{classes[i]}: {probabilities[i] * 100:.2f}%")

if __name__ == "__main__":
    main()
