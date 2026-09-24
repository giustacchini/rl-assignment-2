import gymnasium as gym

from dqn import DQN

env = gym.make("LunarLander-v3", render_mode="human")

state, info = env.reset()

# Instantiate the DQN network
dqn = DQN(hidden_layers=[16, 16], target_update_frequency=100)
batch_size = 64

for _ in range(500):
    # action = env.action_space.sample()
    action = dqn.choose_action(state)

    current_state = state
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    dqn.update_experience_replay(
        current_state,
        action,
        reward,
        next_state,
        done,
    )
    dqn.replay(batch_size)

    state = next_state

    if done:
        state, info = env.reset()

env.close()
