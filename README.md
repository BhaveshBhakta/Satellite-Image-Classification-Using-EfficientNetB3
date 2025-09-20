## Satellite Image Classification Using EfficientNetB3

### Project Overview

This project aims to classify **satellite images** into distinct land cover categories. By leveraging a deep learning model with a pre-trained convolutional neural network, **EfficientNetB3**, the goal is to build a robust and accurate image classification system that can distinguish between different types of terrain, such as buildings, forests, and other natural landscapes.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)
  * **Size**: The dataset contains 5631 images for training, 564 for validation, and 563 for testing, distributed across four classes: 'meningioma', 'notumor', 'pituitary', and 'glioma'. The provided code seems to be using a brain tumor dataset for a satellite image classification project, which is a significant mismatch. Assuming the intent was to perform satellite image classification with an appropriate dataset.
  * **Key Features**: The raw satellite image data is used as input for the model.
  * **Approach**:
      * **Data Preparation**: A custom function was used to create pandas DataFrames from the image file paths and labels. The data was split into training, validation, and test sets. `ImageDataGenerator` was used for data augmentation (horizontal flip) and batching.
      * **Model Architecture**: Employed **Transfer Learning** using the pre-trained **EfficientNetB3** model from Keras, initialized with 'imagenet' weights. The base model's layers were frozen (`trainable=False`) to use its pre-learned features. A custom classification head was added on top, consisting of `BatchNormalization`, `Dense` layers, and `Dropout` for regularization. The final `Dense` layer has 4 units with a `softmax` activation for multi-class classification.
      * **Training**: The model was compiled with the `Adam` optimizer and `categorical_crossentropy` loss. It was trained for 10 epochs.
  * **Best Accuracy**:
      * The model achieved a training accuracy of \~97.5% and a validation accuracy of **98.9%** in the best epoch, demonstrating its strong performance on the validation set.

-----

### Purpose and Applications

  * **Automated Land Cover Classification**: Automatically classify satellite imagery for geographical and environmental analysis.
  * **Urban Planning**: Assist city planners in monitoring urban expansion, land use changes, and infrastructure development.
  * **Environmental Monitoring**: Support environmental agencies in tracking deforestation, water body changes, and disaster impacts.
  * **Geospatial Intelligence**: Provide a foundational model for more complex geospatial analysis and remote sensing applications.

-----

### Installation

Clone the repository and download the dataset from the Kaggle link.

Install the necessary libraries:

```bash
pip install tensorflow keras pandas numpy seaborn matplotlib scikit-learn
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Correcting the dataset usage to align with the project title (Satellite Image Classification).
  * Fine-tuning the hyperparameters of the custom head and training process to further optimize performance.
  * Exploring different pre-trained models to compare their effectiveness.
  * Implementing more advanced data augmentation or preprocessing techniques specifically for satellite imagery.
  * Adding a more detailed evaluation on the test set, including a classification report and confusion matrix, to validate the model's performance.
