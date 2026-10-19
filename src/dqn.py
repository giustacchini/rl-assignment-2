# python
# DQN.py

import random

import keras
import numpy as np
import tensorflow as tf


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
        tau: float = 0.005,
        gamma: float = 0.99,
        replay_capacity: int = 100,
        learning_rate: float = 0.001,
        epsilon_min: float = 0.05,
        epsilon: float = 1.0,
        decay_episodes: int = 1000,
    ):
        """
        Initialize the DQN model with the specified hidden layers and target network update frequency.
        :param hidden_layers: A list of integers specifying the number of neurons in each hidden layer.
        :param tau: The target-network Polyak averaging coefficient.
        """
        if hidden_layers is None:
            hidden_layers = [16]

        self.gamma = gamma
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.epsilon_decay = (epsilon_min / epsilon) ** (1 / decay_episodes)
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon

        self.network = keras.Sequential()
        self.network.add(keras.Input(shape=(8,)))
        for layer in hidden_layers:
            self.network.add(keras.layers.Dense(layer))
            self.network.add(keras.layers.ReLU())
        self.network.add(keras.layers.Dense(4))
        self.network.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
            loss=keras.losses.Huber(),
        )

        self.target_network = keras.models.clone_model(self.network)
        self.target_network.set_weights(self.network.get_weights())
        self.tau = tau
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
        """Move target-network weights toward online-network weights."""
        for target_variable, online_variable in zip(
            self.target_network.weights,
            self.network.weights,
        ):
            target_variable.assign(
                self.tau * online_variable
                + (1.0 - self.tau) * target_variable
            )

    def update_experience_replay(self, state, action, reward, next_state, terminated):
        """
        Store one transition in the experience replay buffer.
        :param state: The state before taking the action.
        :param action: The action taken in the state.
        :param reward: The reward received after taking the action.
        :param next_state: The state resulting from the action.
        :param terminated: Whether the environment reached a terminal state.
        """
        if len(self.experience_replay) < self.replay_capacity:
            self.experience_replay.append((state, action, reward, next_state, terminated))
        else:
            self.experience_replay.pop(0)
            self.experience_replay.append((state, action, reward, next_state, terminated))

    def sample_experiences(self, batch_size):
        """Randomly sample a batch of transitions from replay memory."""
        if len(self.experience_replay) < batch_size:
            raise ValueError("Not enough experiences to sample this batch")

        batch = random.sample(self.experience_replay, batch_size)
        states, actions, rewards, next_states, terminated_s = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(terminated_s, dtype=np.float32),
        )

    def train(self, states, actions, rewards, next_states, terminated_s, episode):
        """
        Train the DQN model using the provided experience replay data and a target network.

        :param states: A batch of states (input to the network).
        :param actions: A batch of actions taken in those states.
        :param rewards: A batch of rewards received after taking those actions.
        :param next_states: A batch of next states resulting from those actions.
        :param terminated_s: A batch indicating which transitions reached terminal states.
        """

        # Convert data to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        terminated_s = tf.convert_to_tensor(terminated_s, dtype=tf.float32)

        # ---------------------------------------------------------
        # 1. Calculate target Q-values using the target network
        # ---------------------------------------------------------
        next_q_values = self.target_network(next_states, training=False)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + (1.0 - terminated_s) * self.gamma * max_next_q_values

        # ---------------------------------------------------------
        # 2. Calculate loss for the online network
        # ---------------------------------------------------------
        with tf.GradientTape() as tape:
            q_values = self.network(states, training=True)

            # Select Q(s,a) for the actions that were actually taken
            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_q_values = tf.gather_nd(q_values, indices)

            # Huber loss
            loss = self.network.loss(targets, selected_q_values)

        # ---------------------------------------------------------
        # 3. Calculate and apply gradients
        # ---------------------------------------------------------
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.network.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        # ---------------------------------------------------------
        # 4. Update target network with Polyak averaging
        # ---------------------------------------------------------
        self.train_steps += 1
        self.update_target_network()

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) for epsilon-greedy action selection.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
