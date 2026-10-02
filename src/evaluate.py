#%%
import gymnasium as gym
import keras 
import numpy as np
import matplotlib.pyplot as plt

#%%

env = gym.make("LunarLander-v3")

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "results" / "checkpoint_30000" / "training_model.keras"

model = keras.models.load_model(
    model_path,
    compile=False
)

# Evaluation settings
successful_landings = 0
num_episodes = 30
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


# %%
# Episode returns plot
episodes = range(1, len(episode_returns) + 1)

plt.figure(figsize=(10, 5))
plt.plot(episodes, episode_returns, linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Episode Returns During Evaluation")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()


# %%
# Fuel consumption plot
episodes = range(1, len(fuel_consumptions) + 1)

plt.figure(figsize=(10, 5))
plt.plot(episodes, fuel_consumptions, linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Fuel Consumption")
plt.title("Fuel Consumption During Evaluation")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()


# %%
# Landing accuracy plot
if landing_errors:
    successful_landings_x = range(1, len(landing_errors) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(successful_landings_x, landing_errors, linewidth=2)

    plt.xlabel("Successful Landing")
    plt.ylabel("Landing Error")
    plt.title("Landing Accuracy During Evaluation")
    plt.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.show()


# %%
# Success rate plot
plt.figure(figsize=(6, 5))

plt.bar(["Success Rate"], [success_rate])

plt.ylabel("Success Rate (%)")
plt.ylim(0, 100)
plt.title("Evaluation Success Rate")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()


# %% 
# Summary table
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.axis("off")

table_data = [
    ["Success Rate", f"{success_rate:.2f}%"],
    ["Average Fuel Consumption", f"{average_fuel:.2f}"],
    [
        "Average Landing Error",
        f"{average_landing_error:.4f}"
        if average_landing_error is not None
        else "N/A (No successful landings)"
    ]
]

table = ax.table(
    cellText=table_data,
    colLabels=["Metric", "Result"],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

for col in range(2):
    table[(0, col)].set_text_props(weight="bold")
    table[(0, col)].set_facecolor("aliceblue")

plt.title("Evaluation Summary", fontsize=13, weight="bold", pad=10)

plt.tight_layout()
plt.show()


env.close()
# %%
