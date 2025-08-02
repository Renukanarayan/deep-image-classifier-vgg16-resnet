
import torch
import argparse
import os
from torchvision import models

def load_checkpoint(arch, checkpoint_dir):
    """Load a model checkpoint and rebuild the model dynamically."""
    
    # Define checkpoint path based on selected architecture
    checkpoint_path = os.path.join(checkpoint_dir, f"{arch}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load the correct model architecture
    if arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, len(checkpoint["class_to_idx"])),
            torch.nn.LogSoftmax(dim=1)
        )
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, len(checkpoint["class_to_idx"])),
            torch.nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    print(f"Model {arch} loaded successfully!")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a trained model checkpoint dynamically.")
    parser.add_argument('--arch', type=str, required=True, choices=['resnet50', 'vgg16'],
                        help="Model architecture to load (resnet50 or vgg16)")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help="Directory where checkpoints are stored")
    
    args = parser.parse_args()

    model = load_checkpoint(args.arch, args.checkpoint_dir)
