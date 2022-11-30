import tensorflow as tf
import numpy as np
import json
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt


model = Sequential(
    [

      # Add an Embedding layer expecting input vocab of size 1000, and
      # output embedding dimension of size 64.
      layers.Embedding(input_dim=1000, output_dim=64),

      # Add a LSTM layer with 128 internal units.
      layers.LSTM(128),

      # Add a Dense layer with 10 units.
      layers.Dense(10),
    ]
)
model.summary()

