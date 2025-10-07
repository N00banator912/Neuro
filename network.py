# Network Handler
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-06

# Imports
import tensorflow as tf
from tensorflow import keras
from keras import layers

class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.common = layers.Dense(hidden_size, activation='relu')
        self.policy = layers.Dense(action_size, activation='softmax')  # actor
        self.value = layers.Dense(1)  # critic

    def call(self, inputs):
        x = self.common(inputs)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value