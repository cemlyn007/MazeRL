def benchmark(run_id: str):
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    import discrete_agent
    import helpers
    import tensorboard_writer
    from discrete_dqns import double_dqn
    from environments import random_environment
    from replay_buffers import fast_prioritised_rb
    from tools import episode_rollout_tool
    from tools.actions_visual_tool import ActionsVisualTool
    import random

    random_state = 816673
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    n_actions = 4
    max_capacity = 5000
    batch_size = 32
    max_steps = 750
    evaluate_max_steps = 201
    max_episodes = 750
    epsilon = 1.0
    delta = 0.0000008
    minimum_epsilon = 0.1
    sampling_eps = 1e-7
    tau = 50  # target network episode update rate
    hps = helpers.Hyperparameters(gamma=0.9, lr=5.0e-4)
    evaluate_reached_goal_count = 0

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    display_game = False
    display_tools = False

    environment = random_environment.RandomEnvironment(
        display=display_game, magnification=500
    )
    environment.draw(environment.init_state)
    dqn = double_dqn.DiscreteDoubleDQN(hps, n_actions, device)
    agent = discrete_agent.DiscreteAgent(environment, dqn, n_actions, stride=0.02)
    rb = fast_prioritised_rb.FastPrioritisedExperienceReplayBuffer(
        max_capacity, batch_size, sampling_eps, agent
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
    }

    def metrics(rewards):
        return {
            "metrics/mean_reward": np.mean(rewards),
            "metrics/min_reward": np.min(rewards),
            "metrics/max_reward": np.max(rewards),
            "metrics/std_reward": np.std(rewards),
            "metrics/median_reward": np.median(rewards),
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

    step_id = 0
    episodes_iter = tqdm(range(max_episodes))
    for episode_id in episodes_iter:
        episode_loss_list = []
        episode_reward_list = []
        agent.reset()
        agent.dqn.train()
        for step_num in range(max_steps):
            transition, distance_to_goal = agent.step(epsilon)
            state, action, reward, next_state = transition
            rb.store(state, action, reward, next_state)
            episode_reward_list.append(reward)

            if len(rb) > batch_size:
                transitions = rb.batch_sample().to(device)
                losses = dqn.train_q_network(transitions)
                episode_loss_list.append(losses.sum())

            if epsilon > minimum_epsilon:
                epsilon -= delta
                epsilon = max(epsilon, minimum_epsilon)
                episodes_iter.set_description(
                    f"Epsilon: {epsilon:.3f} | Goal Count: {evaluate_reached_goal_count}"
                )

            if dqn.HAS_TARGET_NETWORK and (step_id % tau == 0):
                dqn.update_target_network()
            step_id += 1

            if distance_to_goal < 0.03:
                break

        agent.dqn.eval()
        agent.reset()
        states = [agent.state]
        has_reached_goal = False
        for _ in range(max_steps):
            transition, distance_to_goal = agent.step(0.0)
            state, action, reward, next_state = transition
            states.append(agent.state)
            rb.store(state, action, reward, next_state)

            if distance_to_goal < 0.03:
                evaluate_reached_goal_count += 1
                has_reached_goal = True
                break

        rewards = np.array(episode_reward_list)
        log("reward", rewards, episode_id)
        writer.add_histogram("reward_dist", rewards, episode_id)
        step_losses = np.array(episode_loss_list)
        log("loss", step_losses, episode_id)
        writer.add_hparams(hyperparameters, metrics(rewards))
        writer.add_scalar("reached_goal", has_reached_goal, episode_id)
        writer.add_scalar("reached_goal_count", evaluate_reached_goal_count, episode_id)
        writer.add_scalar("epsilon", epsilon, episode_id)

        rollout_tool.set_states(np.asarray(states[:evaluate_max_steps]))
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

    for i in range(int(sys.argv[2])):
        target = functools.partial(benchmark, f"{sys.argv[1]}_{i}")
        process = multiprocessing.Process(target=target)
        start = time.monotonic()
        process.start()
        process.join()
        end = time.monotonic()
        print(f"Iteration {i} took {end - start:.2f} seconds")
