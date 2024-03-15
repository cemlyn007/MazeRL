def benchmark(run_id: str):
    import warnings
    warnings.filterwarnings("error")
    import cv2
    import numpy as np
    import torch

    import helpers
    import tensorboard_writer
    import jax_discrete_dqns.double_dqn
    from environments import random_environment, jax_random_environment
    import environments.renderer
    from jax_replay_buffers import fast_prioritised_rb
    from tools import episode_rollout_tool
    from tools.actions_visual_tool import ActionsVisualTool
    import random
    import jax
    import jax.numpy as jnp
    from flax.core import scope

    class GetQValues:
        def __init__(self, q_network: jax_discrete_dqns.double_dqn.StatelessDiscreteDoubleDQN):
            self.q_network = q_network
            self._predict_q_values = jax.jit(self.q_network.predict_q_values)
            self._network_state = None

        def __call__(self, observations: np.ndarray) -> np.ndarray:
            return np.asarray(self._predict_q_values(self._network_state, jnp.asarray(observations)))
        
        def update_q_network(self, network: scope.VariableDict):
            self._network_state = network

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

    environment = jax_random_environment.RandomEnvironment()
    renderer = environments.renderer.EnvironmentRenderer("RandomEnvironment", 500, np.asarray(environment.free_blocks), np.asarray(environment.goal_state))
    init_state = np.asarray(environment.reset())
    renderer.draw(init_state)
    dqn = jax_discrete_dqns.double_dqn.StatelessDiscreteDoubleDQN(hps, n_actions)

    @jax.jit
    def get_greedy_action(network: scope.VariableDict, observation: jax.Array) -> jax.Array:
        q_values = dqn.predict_q_values(network, jnp.expand_dims(observation, axis=0))
        return jnp.argmax(q_values)
    
    @jax.jit
    def select_action(network: scope.VariableDict, observation: jax.Array, key: jax.Array, epsilon: float) -> jax.Array:
        random_action_key, epsilon_key = jax.random.split(key)
        use_random = jax.random.uniform(epsilon_key) <= epsilon
        random_action = jax.random.randint(random_action_key, (), 0, n_actions)
        greedy_action = get_greedy_action(network, observation)
        return jnp.where(use_random, random_action, greedy_action)
    
    def create_actions(n_actions: int, stride: float) -> jax.Array:
        actions = []
        for i in range(n_actions):
            theta = 2. * np.pi * i / n_actions
            action = stride * np.array([np.cos(theta), np.sin(theta)])
            actions.append(helpers.trunc(action, 5))
        return jnp.stack(actions)
    
    jax_actions = create_actions(n_actions, 0.02)
    actions = np.asarray(jax_actions)

    dqn_state = dqn.reset()
    
    rb = fast_prioritised_rb.StatelessFastPrioritisedExperienceReplayBuffer(
        max_capacity, batch_size, sampling_eps, init_state.shape
    )
    rb_state = rb.reset()
    store = jax.jit(rb.store)
    batch_sample = jax.jit(rb.batch_sample)
    update_batch_weights = jax.jit(rb.update_batch_weights)

    rollout_tool = episode_rollout_tool.EpisodeRolloutTool(renderer.image)
    get_q_values = GetQValues(dqn)
    actions_tool = ActionsVisualTool(500, 15, n_actions, get_q_values)

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

        environment_step = jax.jit(environment.step)

        # We pad the buffer with random transitions to ensure we don't trigger recompilation.
        observation = environment.reset()
        for _ in range(batch_size):
            discrete_action = np.random.randint(0, n_actions)
            next_observation, distance_to_goal = environment_step(
                observation,
                actions[discrete_action]
            )
            reward =  - distance_to_goal
            done = distance_to_goal < 0.03
            rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
            if done:
                observation = environment.reset()
            else:
                next_observation = observation

        train_q_network = jax.jit(dqn.train_q_network)

        @jax.jit
        def step(dqn_state, rb_state, observation, key, epsilon) -> tuple[jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array, jax.Array]:
            select_action_key, sample_key = jax.random.split(key)
            discrete_action = select_action(dqn_state.network, observation, select_action_key, epsilon)
            action = jax_actions[discrete_action]
            next_observation, distance_to_goal = environment_step(
                observation,
                action
            )
            reward = - distance_to_goal
            done = distance_to_goal < 0.03
            rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
            batch, indices = batch_sample(rb_state, sample_key)
            dqn_state, losses = train_q_network(dqn_state, batch)
            rb_state = update_batch_weights(rb_state, indices, losses)
            return dqn_state, rb_state, next_observation, reward, done, losses.sum()
        
        @jax.jit
        def evaluate_step(network, rb_state, observation) -> tuple[fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]:
            discrete_action = get_greedy_action(network, observation)
            next_observation, distance_to_goal = environment_step(
                observation,
                jax_actions[discrete_action],
            )
            reward = - distance_to_goal
            done = distance_to_goal < 0.03
            rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
            return rb_state, next_observation, reward, done

        step_id = 0
        episode_loss_list = []
        episode_reward_list = []
        for episode_id in range(max_episodes):
            episode_loss_list.clear()
            episode_reward_list.clear()
            observation = environment.reset()
            for _ in range(max_steps):
                dqn_state, rb_state, observation, reward, done, loss = step(
                    dqn_state,
                    rb_state,
                    observation,
                    jax.random.PRNGKey(step_id),
                    epsilon
                )
                episode_reward_list.append(reward.item())
                episode_loss_list.append(loss.item())

                if epsilon > minimum_epsilon:
                    epsilon -= delta
                    epsilon = max(epsilon, minimum_epsilon)

                if dqn.has_target_network and (step_id % tau == 0):
                    dqn_state = dqn_state._replace(target_network=dqn_state.network)
                step_id += 1

                if done:
                    break

            observation = environment.reset()
            states = [observation]
            has_reached_goal = False
            for _ in range(evaluate_max_steps):
                rb_state, observation, reward, done = evaluate_step(
                    dqn_state.network,
                    rb_state,
                    observation
                )
                states.append(observation)

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

            get_q_values.update_q_network(dqn_state.network)            
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
