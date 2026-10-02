import json
import matplotlib.pyplot as plt

metrics_path = "results/final_metrics.json"

with open(metrics_path, "r") as file:
    metrics = json.load(file)

episode_returns = metrics["episode_returns"]

episodes = range(1,len(episode_returns)+1)

plt.figure(figsize=(10, 5))

plt.plot(episodes, episode_returns, linewidth=2)
plt.fill_between(episodes, episode_returns, alpha=0.15)

plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Training Learning Curve")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()