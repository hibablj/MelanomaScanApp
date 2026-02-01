# Melanoma-Skin-Cancer-Detection-main

## Introduction

### Melanoma Skin Cancer Detection Project

This project aims to develop an image classification system to detect melanoma, a severe form of skin cancer. The goal is to automatically distinguish benign from malignant lesions using a deep learning model.

The application integrates a Flask-based web interface that allows users to upload images and receive real-time predictions while storing results in a MySQL database. Key technologies include a Convolutional Neural Network (CNN), data augmentation, and REST APIs for statistical analysis.

## Methodology

### Data Preprocessing

- **Data Source**: Images stored in a structured directory  
  - `dataset/benign`  
  - `dataset/malignant`

- **Image Resizing**: Images are resized to 128 × 128 pixels

- **Normalization**: Pixel values are scaled between 0 and 1

- **Data Augmentation**: Applied using `ImageDataGenerator` to reduce overfitting  
  - Rotation  
  - Shifting  
  - Zooming  
  - Other transformations  

### Model Architecture

A sequential CNN model is built with the following layers:

#### Convolutional Layers
- Conv2D(32, (3,3)) + MaxPooling2D  
- Conv2D(64, (3,3)) + MaxPooling2D  
- Conv2D(128, (3,3)) + MaxPooling2D  

#### Dense Layers
- Flatten  
- Dense(128, activation='relu')  
- Dropout(0.5)  
- **Output Layer**: Dense(1, activation='sigmoid') for binary classification  

### Training

- **Loss Function**: binary_crossentropy  
- **Metric**: Accuracy  
- **Parameters**:  
  - 20 epochs  
  - Batch size of 32  
  - 20% validation split  

## Web Integration

- **Flask**: Manages routes for the user interface, predictions, and statistical APIs  
- **Database**: MySQL database storing patient information (name, gender, test results, etc.)

### Features
- Image upload and preview  
- Dashboard displaying patient records and results  
- REST APIs for statistical insights  

## Results

### Model Performance
- Validation accuracy evaluated using `model.evaluate()`  
- Learning curves visualized using `plot_training_curves` and saved as  
  `static/training_curves.png`

### Application Features

- **User Interface**
  - Real-time image upload and prediction  
  - Text-based results (Benign or Malignant)

- **Dashboard**
  - Patient list with detailed records and test outcomes

- **Statistical APIs**
  - Gender-based distribution  
  - Age-based distribution  
  - Counts of positive and negative test results
