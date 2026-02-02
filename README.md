# T-Net-CNN

A Transformer Network with Convolutional Neural Network architecture enhanced with Explainable AI (XAI) capabilities.

## Features

### Core Architecture
- **Hybrid T-Net-CNN Model**: Combines the power of Transformer networks with Convolutional Neural Networks for enhanced feature extraction and pattern recognition
- **Multi-scale Feature Extraction**: Leverages CNN layers for local feature detection and Transformer blocks for global context understanding
- **Attention Mechanisms**: Self-attention and cross-attention modules for capturing long-range dependencies in data

### Explainable AI (XAI) Integration
- **Model Interpretability**: Built-in explainability tools to understand model decisions and predictions
- **Attention Visualization**: Visualize attention weights to see which parts of the input the model focuses on
- **Feature Attribution**: Identify and visualize the most important features contributing to predictions
- **Saliency Maps**: Generate heatmaps highlighting regions of interest in input data

### Training & Optimization
- **Advanced Training Pipeline**: Efficient training workflow with support for distributed training
- **Multiple Loss Functions**: Support for various loss functions tailored to different tasks
- **Learning Rate Scheduling**: Adaptive learning rate strategies for optimal convergence
- **Early Stopping & Checkpointing**: Automatic model saving and early stopping to prevent overfitting

### Data Processing
- **Data Augmentation**: Built-in augmentation techniques to improve model generalization
- **Flexible Input Pipeline**: Support for various data formats and preprocessing options
- **Batch Processing**: Efficient batch processing for training and inference

### Evaluation & Metrics
- **Comprehensive Metrics**: Multiple evaluation metrics including accuracy, precision, recall, F1-score
- **Performance Visualization**: Training curves, confusion matrices, and ROC curves
- **Cross-validation Support**: K-fold cross-validation for robust model evaluation

### Model Management
- **Model Serialization**: Save and load trained models easily
- **Version Control**: Track different model versions and experiments
- **Export Capabilities**: Export models for deployment in various formats

### Utilities
- **Configuration Management**: YAML/JSON-based configuration for easy experimentation
- **Logging & Monitoring**: Comprehensive logging of training progress and metrics
- **GPU Acceleration**: Full CUDA support for faster training and inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/T-Net-CNN.git
cd T-Net-CNN

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Example usage
from tnet_draft_with_xai import TNetCNN

# Initialize model
model = TNetCNN(config='config.yaml')

# Train model
model.train(train_data, val_data)

# Make predictions
predictions = model.predict(test_data)

# Generate explanations
explanations = model.explain(test_data)
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- scikit-learn
- CUDA (optional, for GPU acceleration)

## Project Structure

```
T-Net-CNN/
├── tNet_draft_with_XAI/    # Main source code
├── data/                    # Dataset directory
├── models/                  # Saved models
├── configs/                 # Configuration files
├── notebooks/              # Jupyter notebooks for experiments
├── tests/                  # Unit tests
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tnetcnn2026,
  title={T-Net-CNN: Explainable Hybrid Architecture},
  author={Your Name},
  year={2026}
}
```

## Acknowledgments

- Built with PyTorch
- Inspired by recent advances in Transformer and CNN architectures
- XAI components based on state-of-the-art interpretability research
