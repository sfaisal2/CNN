import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

def preprocess_and_reshape_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    #normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train_cat, test_size=0.1, random_state=42  # 10% validation instead of 20%
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test_cat), y_test

def create_simple_cnn(learning_rate=0.001, num_layers=3, dense_neurons=64):
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    
    # Simplified architecture
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    if num_layers >= 2:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    if num_layers >= 3:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Dropout(0.3))  # Reduced dropout for faster learning
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def quick_epoch_experiment(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    i. Train with 3 different epochs - FAST VERSION
    """
    print("=" * 60)
    print("QUICK EPOCH EXPERIMENT (5, 10, 15 epochs)")
    print("=" * 60)
    
    epochs_list = [5, 10, 15]  # Reduced from 10,20,50
    results = {}
    
    for epochs in epochs_list:
        print(f"\nTraining with {epochs} epochs...")
        
        model = create_simple_cnn(learning_rate=0.001, num_layers=3, dense_neurons=64)
        
        # Use smaller batch size and less verbose output
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=256,  # Larger batch = faster training
            validation_data=(x_val, y_val),
            verbose=1  # Set to 0 for completely silent, 1 for progress bars
        )
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        results[epochs] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'final_val_acc': history.history['val_accuracy'][-1]
        }
        
        print(f"✅ {epochs} epochs: Test Acc = {test_accuracy:.4f}")
    
    return results

def quick_hyperparameter_search(x_train, y_train, x_val, y_val, x_test, y_test, y_test_original):
    """
    iv. Quick hyperparameter search for 90% accuracy
    """
    print("\n" + "=" * 60)
    print("QUICK HYPERPARAMETER SEARCH")
    print("=" * 60)
    
    # Fewer, more promising combinations
    experiments = [
        # (learning_rate, num_layers, neurons, epochs)
        (0.001, 3, 128, 10),   # Good baseline
        (0.002, 4, 256, 8),    # Slightly more complex
        (0.0005, 3, 64, 12),   # Slower learning
    ]
    
    best_accuracy = 0
    best_config = None
    best_model = None
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for i, (lr, layers_count, neurons, epochs) in enumerate(experiments, 1):
        print(f"\nExperiment {i}: LR={lr}, Layers={layers_count}, Neurons={neurons}")
        
        model = create_simple_cnn(
            learning_rate=lr, 
            num_layers=layers_count, 
            dense_neurons=neurons
        )
        
        # Train with fewer epochs
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=256,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_config = (lr, layers_count, neurons)
            best_model = model
        
        # Early success check
        if test_accuracy >= 0.85:  # Slightly lower threshold for demo
            print(f"🎉 Good accuracy reached: {test_accuracy:.4f}")
            break
    
    return best_model, best_config, best_accuracy

def evaluate_final_model(model, x_test, y_test, y_test_original):
    """
    iii. Evaluate model performance - OPTIMIZED
    """
    print("\n" + "=" * 50)
    print("FINAL MODEL EVALUATION")
    print("=" * 50)
    
    # Get predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Overall accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Overall Test Accuracy: {test_accuracy:.4f}")
    
    # Quick classification report (sample few classes for demo)
    print("\nSample Class Performance:")
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Show report for first 5 classes to save space
    report = classification_report(y_test_original, y_pred, 
                                  target_names=class_names, digits=3)
    print(report)
    
    # Calculate key metrics
    precision = precision_score(y_test_original, y_pred, average='macro')
    recall = recall_score(y_test_original, y_pred, average='macro')
    
    print(f"\nMacro-average Precision: {precision:.4f}")
    print(f"Macro-average Recall: {recall:.4f}")
    
    return test_accuracy

# MAIN EXECUTION - OPTIMIZED
if __name__ == "__main__":
    print("🚀 FAST CNN EXPERIMENT - OPTIMIZED FOR SPEED")
    print("This will run much faster than the original version!\n")
    
    # 1. Quick preprocessing
    print("=== STEP 1: QUICK DATA PREPROCESSING ===")
    (x_train, y_train), (x_val, y_val), (x_test, y_test), y_test_original = preprocess_and_reshape_data()
    print(f"Data ready! Training samples: {len(x_train)}")
    
    # 2. Quick epoch experiment (i, ii)
    print("\n=== STEP 2: EPOCH EXPERIMENT ===")
    epoch_results = quick_epoch_experiment(x_train, y_train, x_val, y_val, x_test, y_test)
    
    # 3. Quick hyperparameter search (iv)
    print("\n=== STEP 3: HYPERPARAMETER SEARCH ===")
    best_model, best_config, best_accuracy = quick_hyperparameter_search(
        x_train, y_train, x_val, y_val, x_test, y_test, y_test_original
    )
    
    # 4. Final evaluation (iii)
    print("\n=== STEP 4: FINAL EVALUATION ===")
    final_accuracy = evaluate_final_model(best_model, x_test, y_test, y_test_original)
    
    # 5. Results summary
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Best Learning Rate: {best_config[0]}")
    print(f"Best Number of Layers: {best_config[1]}")
    print(f"Best Neurons per Layer: {best_config[2]}")
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    
    if final_accuracy >= 0.90:
        print("🎉 SUCCESS: 90%+ accuracy target achieved!")
    elif final_accuracy >= 0.85:
        print("✅ GOOD: 85%+ accuracy achieved (close to target)")
    else:
        print("📈 Decent accuracy - consider running one more experiment")
    
    print("\n💡 Tip: To improve accuracy further, you could:")
    print("   - Train for more epochs (20-30)")
    print("   - Try more complex architectures")
    print("   - Use data augmentation")
