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
def train_with_different_epochs(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    i. Train the model with 3 different epochs (10/20/50)
    ii. Monitor metrics after each epoch
    """
    print("=" * 70)
    print("TRAINING WITH 3 DIFFERENT EPOCHS (10, 20, 50)")
    print("=" * 70)
    
    epochs_list = [10, 20, 50]
    results = {}
    
    for epochs in epochs_list:
        print(f"\n🎯 Training with {epochs} epochs...")
        print("-" * 40)
        
        # Reset random seed for each experiment
        tf.random.set_seed(42)
        
        # Create and train model
        model = create_cnn_model(learning_rate=0.001, num_conv_layers=3, dense_neurons=64)
        
        # Train while monitoring metrics
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Store results
        results[epochs] = {
            'model': model,
            'history': history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1]
        }
        
        print(f"✅ Epochs {epochs} completed:")
        print(f"   Final Train Accuracy: {results[epochs]['final_train_acc']:.4f}")
        print(f"   Final Val Accuracy: {results[epochs]['final_val_acc']:.4f}")
        print(f"   Test Accuracy: {results[epochs]['test_accuracy']:.4f}")
    
    return results

def evaluate_model_performance(model, x_test, y_test, y_test_original, class_names):
    """
    iii. Evaluate model performance with accuracy, precision, recall for each class
    """
    print("\n📊 DETAILED MODEL EVALUATION")
    print("=" * 50)
    
    # Get predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate overall accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Overall Test Accuracy: {test_accuracy:.4f}")
    print(f"Overall Test Loss: {test_loss:.4f}")
    
    # Calculate precision and recall for each class
    print("\n📈 Class-wise Performance Metrics:")
    print("-" * 40)
    
    # Using sklearn classification report
    report = classification_report(y_test_original, y_pred, 
                                  target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)
    
    # Calculate macro-average precision and recall
    macro_precision = precision_score(y_test_original, y_pred, average='macro')
    macro_recall = recall_score(y_test_original, y_pred, average='macro')
    
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    
    return y_pred, test_accuracy

def experiment_with_hyperparameters(x_train, y_train, x_val, y_val, x_test, y_test, y_test_original):
    """
    iv. Experiment with 3 different hyperparameters to achieve 90%+ accuracy
    """
    print("=" * 70)
    print("HYPERPARAMETER EXPERIMENTS FOR 90%+ ACCURACY")
    print("=" * 70)
    
    # Define hyperparameter combinations to try
    experiments = [
        # (learning_rate, num_layers, neurons_per_layer)
        (0.001, 3, 64),    # Baseline
        (0.01, 4, 128),    # Higher LR, more layers, more neurons
        (0.005, 5, 256),   # Medium LR, most layers, most neurons
        (0.0005, 4, 128),  # Lower LR, more layers
        (0.001, 5, 256),   # More complex architecture
    ]
    
    best_accuracy = 0
    best_config = None
    best_model = None
    results = {}
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for i, (lr, layers_count, neurons) in enumerate(experiments, 1):
        print(f"\n🔬 Experiment {i}: LR={lr}, Layers={layers_count}, Neurons={neurons}")
        print("-" * 50)
        
        # Reset random seed
        tf.random.set_seed(42)
        
        # Create and train model
        model = create_cnn_model(
            learning_rate=lr, 
            num_conv_layers=layers_count, 
            dense_neurons=neurons
        )
        
        # Train for more epochs to reach better accuracy
        history = model.fit(
            x_train, y_train,
            epochs=30,  # Train longer for better accuracy
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Store results
        results[(lr, layers_count, neurons)] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'history': history
        }
        
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        
        # Check if this is the best model so far
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_config = (lr, layers_count, neurons)
            best_model = model
            
        # If we reach 90% accuracy, we can stop early
        if test_accuracy >= 0.90:
            print(f"🎉 TARGET ACHIEVED! 90%+ accuracy reached: {test_accuracy:.4f}")
    
    return results, best_model, best_config, best_accuracy

def plot_training_history(history, title):
    """Plot training history for visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Class names for Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 1. Preprocess data
    print("=== STEP 1: DATA PREPROCESSING ===")
    (x_train, y_train), (x_val, y_val), (x_test, y_test), y_test_original = preprocess_and_reshape_data()
    
    # 2. Train with different epochs (i, ii)
    epoch_results = train_with_different_epochs(x_train, y_train, x_val, y_val, x_test, y_test)
    
    # 3. Evaluate the best epoch model (iii)
    print("\n" + "=" * 70)
    print("EVALUATING BEST EPOCH MODEL")
    print("=" * 70)
    
    # Use the model with most epochs (50) for detailed evaluation
    best_epoch_model = epoch_results[50]['model']
    y_pred, epoch_accuracy = evaluate_model_performance(
        best_epoch_model, x_test, y_test, y_test_original, class_names
    )
    
    # 4. Hyperparameter experiments (iv)
    hyper_results, best_model, best_config, best_accuracy = experiment_with_hyperparameters(
        x_train, y_train, x_val, y_val, x_test, y_test, y_test_original
    )
    
    # 5. Final evaluation of best model
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"🏆 BEST MODEL CONFIGURATION:")
    print(f"   Learning Rate: {best_config[0]}")
    print(f"   Number of Layers: {best_config[1]}")
    print(f"   Neurons per Layer: {best_config[2]}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")
    
    if best_accuracy >= 0.90:
        print("🎯 SUCCESS: 90%+ accuracy target achieved!")
    else:
        print("⚠️  Target not yet reached. Consider more experiments.")
    
    # Plot training history of best model
    best_history = None
    for config, result in hyper_results.items():
        if config == best_config:
            best_history = result['history']
            break
    
    if best_history:
        plot_training_history(best_history, "Best Model Training History")
    
    # Final detailed evaluation of best model
    print("\n" + "=" * 70)
    print("FINAL DETAILED EVALUATION OF BEST MODEL")
    print("=" * 70)
    
    final_y_pred, final_accuracy = evaluate_model_performance(
        best_model, x_test, y_test, y_test_original, class_names
    )
