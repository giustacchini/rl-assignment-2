# python
# DQN.py

import random

import keras
import numpy as np

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

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        target_update_frequency: int = 100,
        gamma: float = 0.99,
        replay_capacity: int = 100,
        learning_rate: float = 0.001,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        epsilon: float = 1.0,
    ):
        """
        Initialize the DQN model with the specified hidden layers and target network update frequency.
        :param hidden_layers: A list of integers specifying the number of neurons in each hidden layer.
        :param target_update_frequency: An integer specifying how often to update the target network (in training steps).
        """
        if hidden_layers is None:
            hidden_layers = [16]

        self.gamma = gamma
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon

        self.network = keras.Sequential()
        self.network.add(keras.Input(shape=(8,)))
        for layer in hidden_layers:
            self.network.add(keras.layers.Dense(layer))
            self.network.add(keras.layers.ReLU())
        self.network.add(keras.layers.Dense(4))
        self.network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.Huber(),
        )

        self.target_network = keras.models.clone_model(self.network)
        self.target_network.set_weights(self.network.get_weights())
        self.target_update_frequency = target_update_frequency
        self.train_steps = 0
        self.experience_replay = []

    def choose_action(self, state):
        """Choose a random action or the action with the highest Q-value."""
        if random.random() < self.epsilon:
            return random.randrange(4)

        state = np.asarray(state, dtype=np.float32)
        state = np.expand_dims(state, axis=0)
        q_values = self.network(state, training=False).numpy()[0]
        return int(np.argmax(q_values))

    def update_target_network(self):
        """Copy the online network weights to the target network."""
        self.target_network.set_weights(self.network.get_weights())

    def update_experience_replay(self, state, action, reward, next_state, done):
        """
        Store one transition in the experience replay buffer.
        :param state: The state before taking the action.
        :param action: The action taken in the state.
        :param reward: The reward received after taking the action.
        :param next_state: The state resulting from the action.
        :param done: Whether the episode ended after the action.
        """
        if len(self.experience_replay) < self.replay_capacity:
            self.experience_replay.append(
                (state, action, reward, next_state, done)
            )
        else:
            self.experience_replay.pop(0)
            self.experience_replay.append(
                (state, action, reward, next_state, done)
            )

    def sample_experiences(self, batch_size):
        """Randomly sample a batch of transitions from replay memory."""
        if len(self.experience_replay) < batch_size:
            raise ValueError("Not enough experiences to sample this batch")

        batch = random.sample(self.experience_replay, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

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
        targets = rewards + (1 - d_t) * self.gamma * max_target_q_values

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

        self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * self.epsilon_decay,
                )
