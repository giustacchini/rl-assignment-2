# lunar_env.py

import pickle as pkl

import gymnasium as gym
import keras

import Logger
from dqn import DQN
from Logger import Logger
from setup import parse_args


def main():
    args = parse_args()

    if args.batch_size > args.replay_capacity:
        raise ValueError("batch-size cannot be larger than replay-capacity")

    logger = Logger()

    render_mode = "human" if args.render else None

    env = gym.make(
        "LunarLander-v3",
        render_mode=render_mode,
    )

    dqn = DQN(
        hidden_layers=args.hidden_layers,
        target_update_frequency=args.target_update_frequency,
        gamma=args.gamma,
        replay_capacity=args.replay_capacity,
        learning_rate=args.learning_rate,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        decay_episodes=args.decay_episodes,
    )

    # Training history
    episode_returns = []
    episode_fuel_consumptions = []
    episode_successes = []
    episode_landing_errors = []

    # Global training progress
    global_step = 0
    initial_episode = 1

    # Resume checkpoint if requested
    if args.resume is not None:
        with open(
            f"{args.resume}/training_state.pkl",
            "rb",
        ) as f:
            loaded = pkl.load(f)

        dqn.network = keras.saving.load_model(f"{args.resume}/training_model.keras")
        dqn.target_network = keras.saving.load_model(f"{args.resume}/target_network.keras")

        dqn.epsilon = loaded["epsilon"]
        dqn.experience_replay = loaded["experience_replay"]
        dqn.train_steps = loaded["train_steps"]

        global_step = loaded["global_step"]

        episode_returns = loaded["episode_returns"]
        episode_fuel_consumptions = loaded["episode_fuel_consumptions"]
        episode_successes = loaded["episode_successes"]
        episode_landing_errors = loaded["episode_landing_errors"]

        # The checkpoint can occur in the middle of an episode.
        # We cannot restore the exact Gymnasium environment state,
        # therefore that episode starts again from env.reset().
        initial_episode = loaded["episode"]

    # -------------------------
    # Episode training loop
    # -------------------------

    for episode in range(initial_episode, args.episodes + 1):
        state, info = env.reset()

        episode_return = 0.0
        episode_fuel_consumption = 0.0

        episode_done = False

        while not episode_done:  # Updates for each step in the episode
            action = dqn.choose_action(state)

            current_state = state

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_done = terminated or truncated
            global_step += 1

            # -------------------------
            # Fuel consumption
            # -------------------------

            if action == 2:
                # Main engine
                episode_fuel_consumption += 0.3

            elif action in (1, 3):
                # Orientation engines
                episode_fuel_consumption += 0.03

            episode_return += reward

            # -------------------------
            # Experience replay
            # -------------------------

            # Store EVERY transition.
            dqn.update_experience_replay(
                current_state,
                action,
                reward,
                next_state,
                terminated,
            )

            # Train once enough experiences exist, and then train every step.
            if len(dqn.experience_replay) >= args.batch_size:
                states, actions, rewards, next_states, terminated_s = dqn.sample_experiences(batch_size=args.batch_size)

                dqn.train(states, actions, rewards, next_states, terminated_s, episode)

            state = next_state

            # print(
            #     f"Episode {episode}: "
            #     f"steps = {global_step}, "
            #     f"return = {episode_return:.2f}, "
            #     # f"average = {average_return:.2f}, "
            #     f"fuel = {episode_fuel_consumption:.2f}, "
            #     f"epsilon = {dqn.epsilon:.4f}"
            # )

            # -------------------------
            # Checkpoint every N steps
            # -------------------------

            if args.checkpoint_frequency > 0 and global_step % args.checkpoint_frequency == 0:
                logger.save_checkpoint(
                    dqn,
                    global_step,
                    episode,
                    episode_returns,
                    episode_fuel_consumptions,
                    episode_successes,
                    episode_landing_errors,
                )

        # -------------------------
        # Episode finished
        # -------------------------
        dqn.decay_epsilon()
        episode_returns.append(float(episode_return))

        episode_fuel_consumptions.append(float(episode_fuel_consumption))

        # Keep the existing success criterion for now.
        successful_landing = reward == 100
        episode_successes.append(successful_landing)

        if successful_landing:
            episode_landing_errors.append(float(abs(next_state[0])))

        average_return = sum(episode_returns[-100:]) / len(episode_returns[-100:])

        logger.print_episode(
            episode,
            global_step,
            episode_return,
            average_return,
            episode_fuel_consumption,
            dqn.epsilon,
        )

    env.close()

    # -------------------------
    # Save final model
    # -------------------------

    logger.save_networks(dqn.network, dqn.target_network)

    # -------------------------
    # Final aggregate metrics
    # -------------------------

    average_fuel_consumption = (
        sum(episode_fuel_consumptions) / len(episode_fuel_consumptions) if episode_fuel_consumptions else 0
    )

    success_rate = sum(episode_successes) / len(episode_successes) if episode_successes else 0

    average_landing_error = (
        sum(episode_landing_errors) / len(episode_landing_errors) if episode_landing_errors else None
    )

    logger.save_final_metrics(
        args,
        global_step,
        episode_returns,
        dqn.epsilon,
        episode_fuel_consumptions,
        average_fuel_consumption,
        episode_successes,
        episode_landing_errors,
        success_rate,
        average_landing_error,
    )


if __name__ == "__main__":
    main()
