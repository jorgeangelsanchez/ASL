"""
Our Old Model bases on the structure used for Project 5
"""

import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

from tensorflow.keras import losses

import hyperparameters as hp


class YourModel(tf.keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = hp.learning_rate)

        
        self.architecture = [
            #Block 1
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),

            # Block 2
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block4_pool"),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(26, activation='softmax')
            ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        scce = losses.sparse_categorical_crossentropy(labels,predictions)
        return scce