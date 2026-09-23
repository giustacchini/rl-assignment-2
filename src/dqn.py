# python
# DQN.py

import keras

GAMMA = 0.99  # Discount factor for future rewards


class DQN:
    """
    Deep Q-Network (DQN) implementation using Keras.

    input: 8-dimensional state space vector
        1. X-coordinate of the lander
        2. Y-coordinate of the lander
        3. X-velocity of the lander
        4. Y-velocity of the lander
        5. Angle of the lander
        6. Angular velocity of the lander
        7. Boolean indicating left leg contact with the ground (0 or 1)
        8. Boolean indicating right leg contact with the ground (0 or 1)
    output: 4-dimensional action space vector
        1. Action 0: Do nothing
        2. Action 1: Fire left orientation engine
        3. Action 2: Fire main engine
        4. Action 3: Fire right orientation engine
    """

    def __init__(self, hidden_layers: list[int] = [16], target_update_frequency: int = 100):
        """
        Initialize the DQN model with the specified hidden layers and target network update frequency.
        :param hidden_layers: A list of integers specifying the number of neurons in each hidden layer.
        :param target_update_frequency: An integer specifying how often to update the target network (in training steps).
        """
        self.network = keras.Sequential()
        self.network.add(keras.Input(shape=(8,)))
        for layer in hidden_layers:
            self.network.add(keras.layers.Dense(layer))
            self.network.add(keras.layers.ReLU())
        self.network.add(keras.layers.Dense(4))

        self.target_network = keras.models.clone_model(self.network)
        self.target_network.set_weights(self.network.get_weights())
        self.target_update_frequency = target_update_frequency
        self.train_steps = 0
        self.experience_replay = []

    def update_target_network(self):
        """Copy the online network weights to the target network."""
        self.target_network.set_weights(self.network.get_weights())

    def train(self, states, actions, rewards, next_states, d_t):
        """
        Train the DQN model using the provided experience replay data and a target network.

        :param states: A batch of states (input to the network).
        :param actions: A batch of actions taken in those states.
        :param rewards: A batch of rewards received after taking those actions.
        :param next_states: A batch of next states resulting from those actions.
        :param d_t: A batch of boolean values indicating if the episode ended after each action.
        """
        # Compute target Q-values
        target_q_values = self.target_network.predict(next_states)
        max_target_q_values = target_q_values.max(axis=1)
        targets = rewards + (1 - d_t) * GAMMA * max_target_q_values

        # Create a mask for the actions taken
        action_masks = keras.utils.to_categorical(actions, num_classes=4)

        # Compute the predicted Q-values for the current states
        predicted_q_values = self.network.predict(states)

        # Update only the Q-values for the actions taken
        predicted_q_values[action_masks.astype(bool)] = targets

        # Train the network on the updated Q-values
        self.network.fit(states, predicted_q_values, epochs=1, verbose=0)

        self.train_steps += 1
        if self.train_steps % self.target_update_frequency == 0:
            self.update_target_network()
