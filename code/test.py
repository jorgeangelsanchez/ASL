import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

test = pd.read_csv("/home/theo_mcarn/sign_mnist_test.csv")
train = pd.read_csv("/home/theo_mcarn/sign_mnist_train.csv")

print(train.shape)
print(test.shape)

train_set = np.array(train, dtype = 'float32')
test_set = np.array(test, dtype = 'float32')

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

X_train = train_set[:, 1:] / 255
y_train = train_set[:, 0]

X_test = test_set[:, 1:] / 255
y_test = test_set[:,0]

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2,
                                                            random_state = 12345)
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

print(X_train.shape)
print(y_train.shape)
print(X_validate.shape)


#Library for CNN Model
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
import hyperparameters as hp

#Defining the Convolutional Neural Network

cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3, 3), input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(64, (3, 3), input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(128, (3, 3), input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 512, activation = 'relu'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(units = 25, activation = 'softmax'))

#Defining the Convolutional Neural Network
cnn_model.summary()
opt = keras.optimizers.Adam(learning_rate=hp.learning_rate)

cnn_model.compile(loss ='sparse_categorical_crossentropy', 
                    optimizer=opt ,metrics =['accuracy'])



#Training the CNN model
history = cnn_model.fit(X_train, 
                        y_train, 
                        batch_size = hp.batch_size, 
                        epochs = hp.num_epochs, 
                        verbose = 1, 
                        validation_data = (X_validate, y_validate))

cnn_model.save("our_model")

#Visualizing the training performance
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_Loss')
plt.legend()
plt.grid()
plt.title('Loss evolution')
plt.savefig("LossEval.png")

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.grid()
plt.title('Accuracy evolution')
plt.savefig("AccuracyEval.png")

plt.show()

# predicted_classes = cnn_model.predict_classes(X_test)

# L = 5
# W = 5

# fig, axes = plt.subplots(L, W, figsize = (12,12))
# axes = axes.ravel()

# for i in np.arange(0, L * W):  
#     axes[i].imshow(X_test[i].reshape(28,28))
#     axes[i].set_title(f"Prediction Class = {predicted_classes[i]:0.1f}\n True Class = {y_test[i]:0.1f}")
#     axes[i].axis('off')
# plt.subplots_adjust(wspace=0.5)




