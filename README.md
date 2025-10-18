# Convolutional Neural Network Model

## Description 
This is a CNN Model to classify grayscale images of clothing into 10 clothing categories using the 70,000 images in the Fashion MNIST Dataset to train and test the model. The model is implemented in Python, using Keras and TensorFlow.   

## Dataset
The clothing items included in the Fashion MNIST, and the categories we classify the images into are:
- T-shirt/top
- trouser
- pullover
- dress
- coat
- sandal
- shirt
- sneaker
- bag
- ankle boot

## Final Architecture
- Learning Rate: 0.001
- Number of Convolution Layers: 3
- Neurons in Final Dense Layer: 128
- Number of Epochs: 10
- Convolutional Layers:
  - Layer 1: 32 filters, (3×3) kernel, ReLU activation
  - Layer 2: 64 filters, (3×3) kernel, ReLU activation + MaxPooling (2×2)
  - Layer 3: 64 filters, (3×3) kernel, ReLU activation
- Pooling Method: MaxPooling2D with (2×2) pool size after first two convolutional blocks
- Dense Layers: One hidden layer with 128 neurons (ReLU) + Dropout (0.5) + Output layer (10 neurons, softmax)

## Training and Evaluation Metrics
Epoch Experiment Results:
- 5 epochs: Test Accuracy = 0.8679
- 10 epochs: Test Accuracy = 0.8963
- 15 epochs: Test Accuracy = 0.9060
Hyperparameter Experiment Results:
1. LR=0.001, Layers=3, Neurons=64: 0.8948
2. LR=0.005, Layers=3, Neurons=64: 0.8973
3. LR=0.001, Layers=4, Neurons=64: 0.8971
4. LR=0.001, Layers=3, Neurons=128: 0.9014 (BEST)
