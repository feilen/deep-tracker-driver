import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

num_tracked_inputs = 3
num_tracked_outputs = 7

neuralnet = keras.Sequential(
    [
        keras.Input(shape=(3, 4, num_tracked_inputs)),
        layers.LSTM(4),
        layers.LSTM(4),
        layers.LSTM(4),
        layers.Dropout(0.2),
        layers.Dense((3, 4, num_tracked_outputs), activation="sigmoid"),
    ],
    name="neuralnet",
)
neuralnet.summary()


