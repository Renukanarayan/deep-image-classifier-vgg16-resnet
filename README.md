# Flower Image Classifier

A PyTorch-based deep learning model that classifies flower species using transfer learning with VGG16/ResNet architectures. Built as the final project for Udacity's AI Programming with Python Nanodegree.

## Features

- **Transfer Learning**: Uses pre-trained VGG16/ResNet models for efficient training
- **Command-line Interface**: Complete CLI tools for training and inference
- **Model Checkpoints**: Save and load trained models
- **GPU Acceleration**: CUDA support for faster training/prediction
- **Top-K Predictions**: Get multiple predictions with confidence scores
- **Flexible Architecture**: Easy to extend with new model architectures

## Quick Start

1. **Install dependencies**
   ```bash
   pip install torch torchvision numpy Pillow matplotlib
   ```

2. **Download dataset**
   ```bash
   wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
   unzip flower_data.zip
   ```

3. **Train model**
   ```bash
   python train.py flowers/ --arch vgg16 --epochs 10 --gpu
   ```

4. **Make predictions**
   ```bash
   python predict.py image.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json
   ```

## Usage

### Training a Model
```bash
python train.py flowers/ --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu
```

**Training Options:**
- `--save_dir`: Directory to save checkpoints (default: current directory)
- `--arch`: Model architecture - `vgg13`, `vgg16`, `resnet18` (default: vgg16)
- `--learning_rate`: Learning rate for optimizer (default: 0.003)
- `--hidden_units`: Hidden layer size (default: 512)
- `--epochs`: Number of training epochs (default: 10)
- `--gpu`: Use GPU for training

### Making Predictions
```bash
python predict.py rose.jpg checkpoints/checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```

**Prediction Options:**
- `--top_k`: Return top K most likely classes (default: 1)
- `--category_names`: JSON file mapping categories to flower names
- `--gpu`: Use GPU for inference

## Dataset Structure

The flowers dataset should be organized as follows:
```
flowers/
├── train/          # Training images (organized by class folders)
├── valid/          # Validation images  
└── test/           # Test images
```

Each subdirectory contains folders numbered 1-102, representing different flower categories.

## Model Architecture

The classifier uses transfer learning by:
1. Loading a pre-trained CNN (VGG/ResNet) trained on ImageNet
2. Freezing the feature extraction layers
3. Replacing the classifier with a custom fully-connected network
4. Training only the new classifier on the flower dataset

This approach achieves high accuracy with minimal training time.

## Performance

- **Training Time**: ~30-45 minutes on GPU, 3-4 hours on CPU
- **Accuracy**: Typically 85-92% on validation set
- **Inference**: ~50ms per image on GPU

## Project Structure
```
flower-image-classifier/
├── train.py              # Training script
├── predict.py            # Prediction script  
├── cat_to_name.json      # Class name mapping (1-102 to flower names)
├── requirements.txt      # Python dependencies
├── flowers/              # Dataset directory
└── ImageClassifier.ipynb  # Development notebook
```

## Example Output

```bash
$ python predict.py test_image.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json

Top 3 predictions:
1. Pink primrose (88.5%)
2. Hard-leaved pocket orchid (7.2%)
3. Canterbury bells (2.1%)
```

## Requirements
- Python 3.7+
- PyTorch, torchvision
- NumPy, Pillow, matplotlib
- 4GB+ RAM for training
- CUDA-compatible GPU (optional, for acceleration)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Notes
- Checkpoint files can be large (100MB+) - avoid committing to git
- GPU training is ~6-8x faster than CPU
- The `cat_to_name.json` file maps category numbers (1-102) to actual flower names

## License
MIT License
