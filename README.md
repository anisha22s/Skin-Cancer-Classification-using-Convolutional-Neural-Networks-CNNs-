# Skin Cancer Detection using Convolutional Neural Networks (CNNs)

Skin Cancer Detection is a predictive model developed using Convolutional Neural Networks (CNNs) to accurately classify and predict different types of skin lesions associated with skin cancer.

## Key Aspects of our Predictive Analytics Approach:

### Model

Our predictive model is based on Convolutional Neural Networks (CNNs), which are particularly well-suited for image classification tasks. CNNs have shown excellent performance in various computer vision applications. By leveraging CNNs, we can effectively learn and extract features from skin lesion images, leading to accurate predictions.

### Dataset

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 
### Informative Features

In the context of our predictive model, the most informative features are the visual patterns and characteristics extracted from the skin lesion images. The CNN model automatically learns and identifies relevant features that contribute to accurate classification. During the training process, the model learns the exact features and patterns that are informative for the task.

### Model Building

In the model building phase, we followed a systematic approach to construct and initialize the model, prepare the data, and define the optimization and loss functions.

1. Model Architecture: We selected an appropriate CNN architecture based on our desired model_name parameter. We had the flexibility to choose from popular architectures such as ResNet, VGG, DenseNet, and Inception. The initialize_model function was used to set up the model, considering factors like feature extraction and whether to use pre-trained weights.

2. Data Transformations: To ensure optimal performance, we employed various data transformations, including resizing the images to the specified input_size, applying random horizontal and vertical flips, random rotation, and color jittering. Additionally, we converted the images to tensors and normalized the pixel values using the computed mean and standard deviation. These transformations facilitated data augmentation and ensured the model's ability to generalize well to unseen data.

3. Dataset Preparation: We implemented a custom PyTorch Dataset class named HAM10000 to prepare the training and validation data. This class incorporated the training and validation data from the respective DataFrames, applying the specified transformations to the images. DataLoader objects were then created to efficiently process the data in batches during model training and evaluation.

4. Model Optimization: The Adam optimizer with a learning rate of 1e-3 was employed to optimize the model. This adaptive optimization algorithm helped update the model parameters efficiently and expedited convergence.

5. Loss Function: We utilized cross-entropy loss, commonly employed in multi-class classification tasks, to calculate the model's loss during the training process and monitor its performance.

### Model Training

The model training phase involved training the model using the prepared training dataset and evaluating its performance on the validation dataset. The goal was to optimize the model's parameters and improve its ability to accurately classify skin lesion images.

We followed a systematic approach consisting of the following steps:

1. Training Function: We defined a training function that took the training data loader, model, criterion (loss function), optimizer, and current epoch as inputs. Within this function, we performed the necessary steps to train the model, including forward and backward propagation, updating model parameters, and monitoring training progress.

2. Validation Function: We implemented a validation function to assess the model's performance on the validation dataset. This function took the validation data loader, model, criterion, optimizer, and current epoch as inputs. It evaluated the model's accuracy and loss on the validation dataset.

3. Training Loop: To train the model, we implemented a training loop that iterated over the specified number of epochs. For each epoch, we called the training function to train the model on the training dataset and obtained the average training loss and accuracy. We then called the validation function to evaluate the model on the validation dataset and obtained the average validation loss and accuracy for that epoch.

### Model Evaluation

The model evaluation phase involved assessing the performance of the trained model on the test dataset and generating a classification report. The classification report provides accuracy, precision, sensitivity, and F1-score metrics for each class, along with the support (number of samples) for each class.

Additionally, an evaluation graph is provided, showing the training and validation metrics over the course of the training process. This graph helps visualize the model's learning progress and performance on unseen validation data.


