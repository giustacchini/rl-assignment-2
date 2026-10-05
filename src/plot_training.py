import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot training metrics.")
parser.add_argument("metrics_path", help="Path to the metrics JSON file")
args = parser.parse_args()

metrics_path = args.metrics_path

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