import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def preprocess():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train_cat, test_size=0.1, random_state=42 
    )
    
    return (x_train_final, y_train_final), (x_val, y_val), (x_test, y_test_cat), y_test

def create_cnn_model(learning_rate=0.001, num_conv_layers=3, dense_neurons=64):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    if num_conv_layers >= 2:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    if num_conv_layers >= 3:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    if num_conv_layers >= 4:
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    
    #dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def epoch_experiment(x_train, y_train, x_val, y_val, x_test, y_test):
    epochs_list = [5, 10, 15]  
    results = {}
    
    for epochs in epochs_list:
        print(f"\nTraining with {epochs} epochs")
        
        model = create_cnn_model(learning_rate=0.001, num_conv_layers=3, dense_neurons=64)
        
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=256,  
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        results[epochs] = test_accuracy
        print(f"{epochs} epochs -> Test Accuracy: {test_accuracy:.4f}")
    
    return results

def hyperparameter_experiment(x_train, y_train, x_val, y_val, x_test, y_test, y_test_original):  
    experiments = [
        # (learning_rate, num_layers, neurons, epochs)
        (0.001, 3, 64, 10),    
        (0.005, 3, 64, 10),     
        (0.001, 4, 64, 10),
        (0.001, 3, 128, 10),
    ]
    
    best_accuracy = 0
    best_config = None
    best_model = None
    
    for i, (lr, num_layers, neurons, epochs) in enumerate(experiments, 1):
        print(f"\nExperiment {i}: LR={lr}, Layers={num_layers}, Neurons={neurons}")
        
        tf.random.set_seed(42)
        model = create_cnn_model(learning_rate=lr, num_conv_layers=num_layers, dense_neurons=neurons)
        
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=256,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_config = (lr, num_layers, neurons)
            best_model = model
    
    return best_model, best_config, best_accuracy

def evaluate_model(model, x_test, y_test, y_test_original):
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Overall Test Accuracy: {test_accuracy:.4f}")
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    #classification report
    print("\nClassification Report:")
    print(classification_report(y_test_original, y_pred, 
                               target_names=class_names, digits=4))
    
    #F1 scores for each class
    f1_scores = f1_score(y_test_original, y_pred, average=None)
    hardest_class_idx = np.argmin(f1_scores)
    hardest_class_name = class_names[hardest_class_idx]
    hardest_f1 = f1_scores[hardest_class_idx]
    
    print(f"\nMOST DIFFICULT CLASS: {hardest_class_name} (F1-score: {hardest_f1:.4f})")
    
    #confusion Matrix
    cm = confusion_matrix(y_test_original, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    #bar graph of f1 scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, f1_scores, color='skyblue')
    bars[hardest_class_idx].set_color('red')
    
    plt.title('F1 Scores by Class (Red = Most Difficult)')
    plt.xlabel('Class Names')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_performance.png', dpi=300, bbox_inches='tight')
    print("Class performance saved as 'class_performance.png'")
    plt.close()
    
    return test_accuracy

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    
    #preprocessing
    print("STEP 1: PREPROCESSING")
    (x_train, y_train), (x_val, y_val), (x_test, y_test), y_test_original = preprocess()
    
    #epoch experiment
    print("\nSTEP 2: EPOCH EXPERIMENT")
    epoch_results = epoch_experiment(x_train, y_train, x_val, y_val, x_test, y_test)
    
    #hyperparameter experiment
    print("\nSTEP 3: HYPERPARAMETER EXPERIMENT")
    best_model, best_config, best_accuracy = hyperparameter_experiment(
        x_train, y_train, x_val, y_val, x_test, y_test, y_test_original
    )
    
    #final evaluation
    print("\nSTEP 4: FINAL EVALUATION")
    final_accuracy = evaluate_model(best_model, x_test, y_test, y_test_original)
    
    print("FINAL RESULTS")
    print(f"Best Learning Rate: {best_config[0]}")
    print(f"Best Number of Layers: {best_config[1]}")
    print(f"Best Neurons per Layer: {best_config[2]}")
    print(f"Final Accuracy: {final_accuracy:.4f}")
    
