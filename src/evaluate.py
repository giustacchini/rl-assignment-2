import gymnasium as gym
import keras 
import numpy as np

env = gym.make("LunarLander-v3")

model = keras.models.load_model(
    "results/20260928_212511/training_model.keras",
    compile=False
)

# Evaluation settings
successful_landings = 0
num_episodes = 100
fuel_consumptions = []
landing_errors = []
episode_returns = []

for episode in range(num_episodes):
    state, info = env.reset()
    total_return = 0
    fuel_consumption = 0

    while True:
        state_input = np.expand_dims(state, axis=0)
        q_values = model(state_input, training=False).numpy()[0]
        action = int(np.argmax(q_values))
        
        if action == 2:
            fuel_consumption += 0.3
        elif action == 1 or action == 3:
            fuel_consumption += 0.03

        next_state, reward, terminated, truncated, info = env.step(action)

        total_return += reward
        state = next_state

        if terminated or truncated:
            episode_returns.append(total_return)
            fuel_consumptions.append(fuel_consumption)
            if reward == 100:
                successful_landings += 1
                landing_errors.append(abs(next_state[0]))
            break

# Calculate evaluation results
average_fuel = np.mean(fuel_consumptions)
success_rate = (successful_landings / num_episodes) * 100

if landing_errors:
    average_landing_error = np.mean(landing_errors)
else:
    average_landing_error = None

print("Evaluation Results")
print(f"Success rate: {success_rate:.2f}%")
print(f"Average fuel consumption: {average_fuel:.2f}")

if average_landing_error is not None:
    print(f"Average landing error: {average_landing_error:.4f}")
else:
    print("Average landing error: No successful landings")

env.close()