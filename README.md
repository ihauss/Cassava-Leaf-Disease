# Cassava Leaf Disease Classification

This project addresses the classification of cassava leaf images into different disease categories using deep learning models. Cassava is a vital crop in many regions, and early detection of leaf diseases can help prevent significant agricultural losses.

## Objective

The goal of this project is to develop a reliable image classification model to identify types of cassava leaf diseases from images. This work is inspired by the Cassava Leaf Disease Classification competition and adapted with a custom pipeline to experiment with different model architectures.

## Dataset

The dataset contains RGB images of cassava leaves labeled with one of the following five classes:

- Cassava Bacterial Blight (CBB)
- Cassava Brown Streak Disease (CBSD)
- Cassava Green Mottle (CGM)
- Cassava Mosaic Disease (CMD)
- Healthy

Due to Kaggle restrictions, the dataset is not included in this repository. Please refer to the [original Kaggle competition page](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) for more information.

## Methodology

The workflow includes the following steps:

- **Data Preprocessing**: Resizing, augmenting, and normalizing images with `torchvision.transforms`.
- **Model Architectures**:
  - ViT (Vision Transformer)
  - MobileNetV3
  - A combined ensemble model that averages predictions from ViT and MobileNetV3.
- **Training**: Conducted with cross-entropy loss, using learning rate scheduling (CosineAnnealingLR).
- **Validation**: Accuracy evaluated on a separate validation set.

## Notable Features

- Custom utility functions for dataset loading and prediction generation.
- Cosine learning rate annealing with configurable `T_max`.
- Use of an ensemble approach to mitigate overfitting and benefit from both large and lightweight models.

## Results

While exact metrics vary by training run, the combination of ViT and MobileNetV3 demonstrated improved generalization on the validation set compared to using either model alone.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- scikit-learn
- OpenCV (for optional visualization)

Install dependencies with:

```bash
pip install -r requirements.txt
