import gymnasium as gym

from dqn import DQN

env = gym.make("LunarLander-v3", render_mode="human")

state, info = env.reset()

# Instantiate the DQN network
dqn = DQN(hidden_layers=[16, 16], target_update_frequency=100)

for _ in range(500):
    action = env.action_space.sample()

    # Use epsilon-greedy policy to select action

    state, reward, terminated, truncated, info = env.step(action)

    # Train the network
    dqn.train(states=state, actions=action, rewards=reward, next_states=state, d_t=terminated or truncated)

    if terminated or truncated:
        state, info = env.reset()

env.close()
