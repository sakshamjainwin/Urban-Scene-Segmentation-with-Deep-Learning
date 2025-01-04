# Urban-Scene-Segmentation-with-Deep-Learning

## Overview
This project implements semantic segmentation on the Cityscapes dataset using four advanced deep learning models:

1. **Deeplabv3+**
2. **PSPNet with Xception Backbone**
3. **U-Net**
4. **Attention Res-UNet**

Each model was trained, evaluated, and compared for performance using metrics like Intersection over Union (IoU), F1 Score, and validation accuracy. The project includes detailed preprocessing, training, and evaluation steps.

---

## Dataset
The [Cityscapes dataset](https://www.cityscapes-dataset.com/) is used for semantic segmentation. It contains high-quality labeled images collected from urban scenes in Germany.

### Key Details:
- **Images:** RGB images.
- **Labels:** Semantic segmentation masks.

---

## Models Implemented

### 1. **Deeplabv3+**
- **Architecture:** DeepLabv3+ is a state-of-the-art model for semantic segmentation, incorporating atrous convolution and encoder-decoder design.
- **Loss Function:** Sparse categorical cross-entropy.
- **Metrics:** Accuracy and Mean IoU.
- **Special Features:**
  - Early stopping.
  - Model checkpointing.
  - Visualization of predictions during training.

### 2. **PSPNet with Xception Backbone**
- **Architecture:** Pyramid Scene Parsing Network (PSPNet) with an Xception backbone.
- **Loss Function:** Sparse categorical cross-entropy.
- **Metrics:** Accuracy and Mean IoU.
- **Special Features:**
  - Pyramid pooling module for global context aggregation.
  - Visualization callbacks for comparing predictions.

### 3. **U-Net**
- **Architecture:** U-Net is a convolutional neural network designed for biomedical image segmentation, adapted here for urban scene segmentation.
- **Loss Function:** Sparse categorical cross-entropy.
- **Metrics:** Accuracy and Mean IoU.
- **Special Features:**
  - Early stopping with patience of 50 epochs.
  - Visualization of training history and predictions.

### 4. **Attention Res-UNet**
- **Architecture:** Combines U-Net with attention mechanisms and residual connections for improved focus on important regions in the image.
- **Loss Function:** Combination of categorical cross-entropy and dice loss (bce_dice_loss).
- **Metrics:** IoU score and F1 score.
- **Special Features:**
  - Data augmentation.
  - Visualization of ground-truth masks and predicted masks.
  - Advanced performance metrics plotted over epochs.

---

## Preprocessing
### Steps:
1. **Resizing:** Images resized to either 128x128 or 256x256.
2. **Normalization:** Pixel values scaled to [0, 1].
3. **Mask Encoding:**
   - RGB masks converted to integer class labels.
   - One-hot encoding for compatibility with loss functions.

---

## Training
### Common Details:
- **Optimizer:** Adam with learning rates between 1e-4 and 1e-3.
- **Batch Sizes:**
  - 8 for Deeplabv3+ and PSPNet.
  - 16 for Attention Res-UNet.
  - 32 for U-Net.
- **Callbacks:** Early stopping, model checkpointing, and custom visualization callbacks.

---

## Evaluation
### Metrics:
1. **IoU (Intersection over Union):** Evaluates the overlap between predicted and ground-truth masks.
2. **F1 Score:** Harmonic mean of precision and recall.
3. **Accuracy:** Proportion of correctly predicted pixels.

### Visualizations:
- Comparison of ground-truth masks and predicted masks.
- Training and validation losses plotted over epochs.
- IoU and F1 scores plotted for evaluation.

---

## Acknowledgments
Special thanks to the creators of the Cityscapes dataset and the deep learning community for their contributions to semantic segmentation research.

