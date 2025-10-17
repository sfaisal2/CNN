from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

#load data 
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

#convert to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#split validation set
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=13)

#reshape all sets
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1) 
X_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#normalize
X_train = X_train.astype('float32')
X_val = X_val.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255
X_val /= 255
X_test /= 255
