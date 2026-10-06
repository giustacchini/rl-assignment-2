from pathlib import Path
import json
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent

metrics_path = (
    project_root
    /"results"
    /"checkpoint_60000"
    /"final_metrics.json"
)

results_dir = metrics_path.parent
plots_dir = results_dir / "plots"
plots_dir.mkdir(exist_ok=True)

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

plt.savefig(
    plots_dir / "training_learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()