# python
# DQN.py

import keras

class DQN:
    def __init__(self, hidden_layers: list[int] = [16]):
        self.network = keras.Sequential()
        self.network.add(keras.Input(shape=(8,)))
        for layer in hidden_layers:
            self.network.add(keras.layers.Dense(layer))
            self.network.add(keras.layers.ReLU())
        self.network.add(keras.layers.Dense(4))