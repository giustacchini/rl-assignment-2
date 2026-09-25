# python
# test_dqn.py

import numpy as np
import pytest

from dqn import DQN

def test_default_dqn_input_layer_matches_state_space_dimensionality():
    dqn = DQN()
    assert dqn.network.layers[0].input.shape == (None, 8)

def test_default_dqn_produces_four_outputs():
    dqn = DQN()
    assert dqn.network.layers[-1].output.shape[1] == 4

def test_default_dqn_accepts_observation_and_produces_expected_q_values():
    import gymnasium as gym
    env = gym.make("LunarLander-v3")

    state, _ = env.reset()

    dqn = DQN()

    # reshape observation to match state space dimensionality
    assert dqn.network(state.reshape((1, 8)))[0].numpy().shape[0] == 4

def test_dqn_hidden_layer_argument_produces_expected_architecture():
    dqn = DQN()
    dqn_custom_hidden_layer = DQN([32, 16])

    assert dqn.network.layers[1].input.shape[1] == 16

    assert dqn_custom_hidden_layer.network.layers[1].input.shape[1] == 32
    assert dqn_custom_hidden_layer.network.layers[3].input.shape[1] == 16


def make_transition(index=0, done=False):
    state = np.full(8, index, dtype=np.float32)
    next_state = np.full(8, index + 1, dtype=np.float32)
    return state, index % 4, float(index), next_state, done


def test_replay_buffer_stores_complete_transition():
    dqn = DQN()
    transition = make_transition()

    dqn.update_experience_replay(*transition)

    assert len(dqn.experience_replay) == 1
    assert len(dqn.experience_replay[0]) == 5
    assert dqn.experience_replay[0][-1] is False


def test_replay_buffer_does_not_exceed_capacity():
    capacity = 4
    dqn = DQN(replay_capacity=capacity)

    for index in range(capacity + 1):
        dqn.update_experience_replay(*make_transition(index))

    assert len(dqn.experience_replay) == capacity
    assert dqn.experience_replay[0][0][0] == 1


def test_dqn_accepts_training_parameters():
    dqn = DQN(gamma=0.95, replay_capacity=32)

    assert dqn.gamma == 0.95
    assert dqn.replay_capacity == 32


def test_sample_experiences_returns_requested_batch_shapes():
    dqn = DQN()
    batch_size = 8

    for index in range(batch_size):
        dqn.update_experience_replay(*make_transition(index))

    states, actions, rewards, next_states, dones = (
        dqn.sample_experiences(batch_size)
    )

    assert states.shape == (batch_size, 8)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, 8)
    assert dones.shape == (batch_size,)


def test_sample_experiences_rejects_undersized_buffer():
    dqn = DQN()

    with pytest.raises(ValueError):
        dqn.sample_experiences(1)


def test_train_trains_and_decays_epsilon():
    dqn = DQN()
    batch_size = 8

    for index in range(batch_size):
        dqn.update_experience_replay(*make_transition(index))

    initial_epsilon = dqn.epsilon
    batch = dqn.sample_experiences(batch_size)

    dqn.train(*batch)

    assert dqn.train_steps == 1
    assert dqn.epsilon < initial_epsilon
    assert dqn.epsilon >= dqn.epsilon_min


def test_sample_experiences_rejects_batch_before_training():
    dqn = DQN()

    with pytest.raises(ValueError):
        dqn.sample_experiences(1)

    assert dqn.train_steps == 0


def test_choose_action_returns_valid_lunar_lander_action():
    dqn = DQN()
    state = np.zeros(8, dtype=np.float32)

    actions = {dqn.choose_action(state) for _ in range(20)}

    assert actions <= {0, 1, 2, 3}