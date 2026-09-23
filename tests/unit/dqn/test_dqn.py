# python
# test_dqn.py

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