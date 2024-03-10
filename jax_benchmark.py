def benchmark(run_id: str):
    import warnings
    warnings.filterwarnings("error")
    import cv2
    import numpy as np
    import torch

    import jax_discrete_agent
    import helpers
    import tensorboard_writer
    import jax_discrete_dqns.double_dqn
    from environments import random_environment
    from jax_replay_buffers import fast_prioritised_rb
    from tools import episode_rollout_tool
    from tools.actions_visual_tool import ActionsVisualTool
    import random
    import jax

    random_state = 816673
    torch.random.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    n_actions = 4
    max_capacity = 5000
    mini_batches = 4
    batch_size = 32 * mini_batches
    if batch_size % mini_batches != 0:
        raise ValueError("Batch size must be a multiple of mini_batches")
    max_steps = 750
    evaluate_max_steps = 201
    max_episodes = 750
    epsilon = 1.0
    delta = 0.0000008
    minimum_epsilon = 0.1
    sampling_eps = 1e-7
    tau = 50  # target network episode update rate
    hps = helpers.JaxHyperparameters(gamma=0.9, lr=5.0e-4, mini_batches=mini_batches)
    evaluate_reached_goal_count = 0

    device = jax.devices("cpu")[0]
    display_game = False
    display_tools = False

    environment = random_environment.RandomEnvironment(
        display=display_game, magnification=500
    )
    environment.draw(environment.init_state)
    dqn = jax_discrete_dqns.double_dqn.DiscreteDoubleDQN(hps, n_actions, device)
    agent = jax_discrete_agent.DiscreteAgent(environment, dqn, n_actions, stride=0.02)
    rb = fast_prioritised_rb.FastPrioritisedExperienceReplayBuffer(
        max_capacity, batch_size, sampling_eps, environment.init_state.shape
    )

    rollout_tool = episode_rollout_tool.EpisodeRolloutTool(environment.renderer.image)
    actions_tool = ActionsVisualTool(500, 15, n_actions, agent)

    hyperparameters = {
        "gamma": hps.gamma,
        "lr": hps.lr,
        "max_capacity": max_capacity,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "max_episodes": max_episodes,
        "initial_epsilon": epsilon,
        "epsilon_decay": delta,
        "minimum_epsilon": minimum_epsilon,
        "random_state": random_state,
        "discrete_actions": True,
        "weighted_replay_buffer": True,
        "sampling_eps": sampling_eps,
        "mini_batches": mini_batches,
    }

    def metrics(rewards) -> dict[str, float]:
        return {
            "metrics/mean_reward": np.mean(rewards).item(),
            "metrics/min_reward": np.min(rewards).item(),
            "metrics/max_reward": np.max(rewards).item(),
            "metrics/std_reward": np.std(rewards).item(),
            "metrics/median_reward": np.median(rewards).item(),
        }

    writer = tensorboard_writer.CustomSummaryWriter(
        log_dir=f"runs/discrete_agent_runs/{run_id}"
    )

    def log(main_tag, values, episode):
        writer.add_scalar(f"{main_tag}/mean", np.mean(values), episode)
        writer.add_scalar(f"{main_tag}/min", np.min(values), episode)
        writer.add_scalar(f"{main_tag}/max", np.max(values), episode)
        writer.add_scalar(f"{main_tag}/std", np.std(values), episode)
        writer.add_scalar(f"{main_tag}/median", np.median(values), episode)

    def log_greedy_policy(draw=True):
        if draw:
            rollout_tool.draw()
        policy_img = cv2.cvtColor(rollout_tool.image, cv2.COLOR_BGR2RGB)
        policy_img = torch.from_numpy(policy_img)
        writer.add_image("greedy_policy", policy_img, episode_id, dataformats="HWC")

    def log_greedy_actions_map(draw=True):
        if draw:
            actions_tool.draw()
        actions_img = cv2.cvtColor(actions_tool.image, cv2.COLOR_BGR2RGB)
        actions_img = torch.from_numpy(actions_img)
        writer.add_image(
            "greedy_actions_map", actions_img, episode_id, dataformats="HWC"
        )

    with jax.log_compiles(True):

        # We pad the buffer with random transitions to ensure we don't trigger recompilation.
        agent.reset()
        state = agent.state
        while len(rb) < batch_size:
            discrete_action = np.random.randint(0, agent._n_actions)
            next_state, distance_to_goal = agent.environment.step(
                state,
                agent._actions[discrete_action]
            )
            reward = agent.compute_reward(distance_to_goal)
            done = distance_to_goal < 0.03
            rb.store(state, discrete_action, reward, done, next_state)
            if done:
                agent.reset()
                state = agent.state
            next_state = state

        step_id = 0
        episode_loss_list = []
        episode_reward_list = []
        for episode_id in range(max_episodes):
            episode_loss_list.clear()
            episode_reward_list.clear()
            agent.reset()
            agent.dqn.train()
            for step_num in range(max_steps):
                transition, distance_to_goal = agent.step(epsilon)
                state, action, reward, next_state = transition
                done = distance_to_goal < 0.03
                rb.store(state, action, reward, done, next_state)
                episode_reward_list.append(reward)

                transitions = rb.batch_sample()
                losses = dqn.train_q_network(transitions)
                rb.update_batch_weights(losses)
                episode_loss_list.append(losses.sum().item())

                if epsilon > minimum_epsilon:
                    epsilon -= delta
                    epsilon = max(epsilon, minimum_epsilon)

                if dqn.has_target_network and (step_id % tau == 0):
                    dqn.update_target_network()
                step_id += 1

                if done:
                    break

            agent.dqn.eval()
            agent.reset()
            states = [agent.state]
            has_reached_goal = False
            for _ in range(evaluate_max_steps):
                transition, distance_to_goal = agent.step(0.0)
                done = distance_to_goal < 0.03
                state, action, reward, next_state = transition
                states.append(agent.state)
                rb.store(state, action, reward, done, next_state)

                if done:
                    evaluate_reached_goal_count += 1
                    has_reached_goal = True
                    break

            if episode_reward_list:
                rewards = np.array(episode_reward_list)
                log("reward", rewards, episode_id)
                writer.add_histogram("reward_dist", rewards, episode_id)
            if episode_loss_list:
                step_losses = np.array(episode_loss_list)
                log("loss", step_losses, episode_id)
            writer.add_hparams(hyperparameters, metrics(rewards))
            writer.add_scalar("reached_goal", has_reached_goal, episode_id)
            writer.add_scalar("reached_goal_count", evaluate_reached_goal_count, episode_id)
            writer.add_scalar("epsilon", epsilon, episode_id)

            rollout_tool.set_states(np.asarray(states))
            if display_tools:
                rollout_tool.draw()
                log_greedy_policy(draw=False)
                rollout_tool.show()
                actions_tool.draw()
                log_greedy_actions_map(draw=False)
                actions_tool.show()
            else:
                log_greedy_policy()
                log_greedy_actions_map()
        print(f"Reached goal: {evaluate_reached_goal_count}")


if __name__ == "__main__":
    import functools
    import multiprocessing
    import time
    import sys

    multiprocessing.set_start_method('spawn')

    for i in range(int(sys.argv[2])):
        target = functools.partial(benchmark, f"{sys.argv[1]}_{i}")
        process = multiprocessing.Process(target=target)
        start = time.monotonic()
        process.start()
        process.join()
        end = time.monotonic()
        print(f"Iteration {i} took {end - start:.2f} seconds")
