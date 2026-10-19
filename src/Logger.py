import json
import os
import pickle as pkl
from datetime import datetime


class Logger:

    def __init__(self):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_location = f"results/{run_timestamp}"
        os.makedirs(self.results_location, exist_ok=True)

    @staticmethod
    def print_episode(self, episode, global_step, episode_return, average_return, episode_fuel_consumption, epsilon):
        run_timestamp = datetime.now().strftime("[%Y-%m-%d][%H:%M:%S]")
        episode_info_str = (
            f"{run_timestamp} "
            f"Episode {episode} - "
            f"Steps = {global_step}, "
            f"Return = {episode_return:.2f}, "
            f"Average_Return = {average_return:.2f}, "
            f"Fuel = {episode_fuel_consumption:.2f}, "
            f"Epsilon = {epsilon:.4f}"
        )
        print(episode_info_str)

        with open(f"{self.results_location}/full_training_log.txt", "a") as f:
            f.write(episode_info_str + "\n")

    def save_checkpoint(
        self,
        dqn,
        global_step,
        episode,
        episode_returns,
        episode_fuel_consumptions,
        episode_successes,
        episode_landing_errors,
    ):
        checkpoint_location = f"{self.results_location}/" f"checkpoint_{global_step}"
        os.makedirs(checkpoint_location, exist_ok=True)

        dqn.network.save(f"{checkpoint_location}/training_model.keras")
        dqn.target_network.save(f"{checkpoint_location}/target_network.keras")

        training_state = {
            "experience_replay": dqn.experience_replay,
            "epsilon": dqn.epsilon,
            "train_steps": dqn.train_steps,
            "global_step": global_step,
            "episode": episode,
            "episode_returns": episode_returns,
            "episode_fuel_consumptions": episode_fuel_consumptions,
            "episode_successes": episode_successes,
            "episode_landing_errors": episode_landing_errors,
        }

        with open(
            f"{checkpoint_location}/training_state.pkl",
            "wb",
        ) as f:
            pkl.dump(training_state, f)

        checkpoint_metrics = {
            "global_step": global_step,
            "episode": episode,
            "epsilon": dqn.epsilon,
            "train_steps": dqn.train_steps,
            "completed_episodes": len(episode_returns),
        }

        with open(
            f"{checkpoint_location}/metrics.json",
            "w",
        ) as f:
            json.dump(checkpoint_metrics, f, indent=4)

    def save_final_metrics(
        self,
        args,
        global_step,
        episode_returns,
        epsilon,
        episode_fuel_consumptions,
        average_fuel_consumption,
        episode_successes,
        episode_landing_errors,
        success_rate,
        average_landing_error,
    ):
        # -------------------------
        # Save final metrics
        # -------------------------

        episode_returns = [float(value) for value in episode_returns]
        episode_fuel_consumptions = [float(value) for value in episode_fuel_consumptions]
        episode_successes = [bool(value) for value in episode_successes]
        episode_landing_errors = [float(value) for value in episode_landing_errors]
        epsilon = float(epsilon)
        average_fuel_consumption = float(average_fuel_consumption)
        success_rate = float(success_rate)
        average_landing_error = (
            None if average_landing_error is None else float(average_landing_error)
        )

        metrics = {
            "configuration": {
                "episodes": args.episodes,
                "batch_size": args.batch_size,
                "hidden_layers": args.hidden_layers,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "epsilon_min": args.epsilon_min,
                "replay_capacity": args.replay_capacity,
                "tau": args.tau,
                "checkpoint_frequency": args.checkpoint_frequency,
            },
            "total_environment_steps": global_step,
            "episode_returns": episode_returns,
            "final_epsilon": epsilon,
            "episode_fuel_consumptions": episode_fuel_consumptions,
            "average_fuel_consumption": average_fuel_consumption,
            "episode_successes": episode_successes,
            "success_rate": success_rate,
            "episode_landing_errors": episode_landing_errors,
            "average_landing_error": average_landing_error,
        }

        metrics_path = f"{self.results_location}/final_metrics.json"
        temporary_metrics_path = f"{metrics_path}.tmp"
        with open(temporary_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        os.replace(temporary_metrics_path, metrics_path)

    def print_episode(self, episode, global_step, episode_return, average_return, episode_fuel_consumption, epsilon):
        run_timestamp = datetime.now().strftime("[%Y-%m-%d][%H:%M:%S]")
        episode_info_str = (
            f"{run_timestamp} "
            f"Episode {episode} - "
            f"steps = {global_step}, "
            f"return = {episode_return:.2f}, "
            f"average_return = {average_return:.2f}, "
            f"fuel = {episode_fuel_consumption:.2f}, "
            f"epsilon = {epsilon:.4f}"
        )
        print(episode_info_str)

        with open(f"{self.results_location}/full_training_log.txt", "a") as f:
            f.write(episode_info_str + "\n")

    def save_networks(self, network, target_network):
        network.save(f"{self.results_location}/training_model.keras")
        target_network.save(f"{self.results_location}/target_network.keras")
