
import os
import argparse
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets

def get_input_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset.")
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'vgg16'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def load_data(data_dir):
    """Load datasets and create dataloaders."""
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False, num_workers=2)
    }

    return dataloaders, image_datasets['train'].class_to_idx, len(image_datasets['train'].classes)

def build_model(arch, hidden_units, num_classes):
    """Build and modify the model for fine-tuning."""
    if arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features  # Fix
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )

    # Freeze all model parameters except classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters() if arch == 'resnet50' else model.classifier[6].parameters():
        param.requires_grad = True

    return model

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, epochs, save_dir, arch):
    """Train the model and save the best checkpoint."""
    model.to(device)
    best_valid_loss = float('inf')

    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Training Phase
        model.train()
        train_loss = 0

        for batch_idx, (images, labels) in enumerate(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            train_loss += loss.item()

            # Print batch update every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloaders['train'])} - Loss: {loss.item():.4f}")

        # Validation Phase
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        valid_accuracy = 100 * correct / total
        print(f"Train Loss: {train_loss/len(dataloaders['train']):.3f}, "
              f"Valid Loss: {valid_loss/len(dataloaders['valid']):.3f}, "
              f"Valid Accuracy: {valid_accuracy:.2f}%")

        # Save best model
        checkpoint = {
            'epoch': epoch + 1,
            'arch': arch,  # Save architecture info
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': dataloaders['train'].dataset.class_to_idx
        }

        save_path = os.path.join(save_dir, f"{arch}.pth")
        torch.save(checkpoint, save_path)
        print(f"Model saved at {save_path} (best validation loss).")

        scheduler.step()

    # Save final model
    final_checkpoint_path = os.path.join(save_dir, f"{arch}_final.pth")
    torch.save(checkpoint, final_checkpoint_path)
    print(f"Final model saved at {final_checkpoint_path}")

    print("Training Complete!")

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, class_to_idx, num_classes = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units, num_classes)

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)

    train_model(model, dataloaders, criterion, optimizer, scheduler, device, args.epochs, args.save_dir, args.arch)

if __name__ == "__main__":
    main()
