import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_and_reshape_data():
    """Preprocess Fashion MNIST data"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encoding for training, keep original for metrics
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    # Train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train_cat, test_size=0.2, random_state=42
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test_cat), y_test

def create_cnn_model(learning_rate=0.001, num_conv_layers=3, dense_neurons=64):
    """
    Create CNN model with customizable hyperparameters
    """
    model = models.Sequential()
    
    # First convolutional block (always present)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional block
    if num_conv_layers >= 2:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Third convolutional block
    if num_conv_layers >= 3:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Fourth convolutional block (for 4+ layers)
    if num_conv_layers >= 4:
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    
    # Fifth convolutional block (for 5 layers)
    if num_conv_layers >= 5:
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile with specific learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
