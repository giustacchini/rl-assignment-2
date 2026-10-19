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

episodes = range(1, len(episode_returns) + 1)
rolling_average = [
    sum(episode_returns[max(0, index - 99): index + 1])
    / len(episode_returns[max(0, index - 99): index + 1])
    for index in range(len(episode_returns))
]

plt.figure(figsize=(10, 5))

plt.plot(episodes, episode_returns, linewidth=2, label="Episode return")
plt.plot(
    episodes,
    rolling_average,
    color="red",
    linewidth=2,
    label="Average return (last 100 episodes)",
)
plt.fill_between(episodes, episode_returns, alpha=0.15)

plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Training Learning Curve")

plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    plots_dir / "training_learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()