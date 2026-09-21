import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

state, info = env.reset()

for _ in range (500):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        state, info = env.reset()

env.close()