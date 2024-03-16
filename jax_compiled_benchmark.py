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

        environment_reset = jax.jit(environment.reset)
        environment_step = jax.jit(environment.step)

        @jax.jit
        def generate_random_rb() -> fast_prioritised_rb.State:

            def body_fun(i: jax.Array, carry: tuple[fast_prioritised_rb.State, jax.Array]) -> tuple[fast_prioritised_rb.State, jax.Array]:
                rb_state, observation= carry
                discrete_action = jax.random.randint(jax.random.PRNGKey(i), (), 0, n_actions)
                next_observation, distance_to_goal = environment_step(
                    observation,
                    jax_actions[discrete_action],
                )
                reward = - distance_to_goal
                done = distance_to_goal < 0.03
                updated_rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
                next_observation = jnp.where(done, environment_reset(), next_observation)
                return updated_rb_state, next_observation

            updated_rb_state, _ = jax.lax.fori_loop(
                0,
                batch_size,
                body_fun,
                (rb.reset(), environment_reset())
            )

            return updated_rb_state

        @jax.jit
        def play_and_train(step_id: jax.Array, dqn_state: jax_discrete_dqns.double_dqn.State, rb_state: fast_prioritised_rb.State, epsilon: jax.Array) -> tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]:

            def cond_fun(carry: tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
                i, step_id, dqn_state, rb_state, next_observation, done, epsilon, episode_rewards, episode_losses = carry
                return jnp.logical_and(i < max_steps, jnp.logical_not(done))

            def body_fun(carry: tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                i, step_id, dqn_state, rb_state, observation, done, epsilon, episode_rewards, episode_losses = carry
                select_action_key, sample_key = jax.random.split(jax.random.PRNGKey(step_id))
                discrete_action = select_action(dqn_state.network, observation, select_action_key, epsilon)
                next_observation, distance_to_goal = environment_step(
                    observation,
                    jax_actions[discrete_action],
                )
                reward = - distance_to_goal
                done = distance_to_goal < 0.03
                updated_rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
                batch, indices = batch_sample(updated_rb_state, sample_key)
                updated_dqn_state, losses = train_q_network(dqn_state, batch)
                weight_updated_rb_state = update_batch_weights(updated_rb_state, indices, losses)

                epsilon = jnp.maximum(epsilon - delta, minimum_epsilon)
                if dqn.has_target_network:
                    update_target = (step_id % tau) == 0
                    updated_dqn_state = updated_dqn_state._replace(target_network=jax.tree_map(
                        lambda x, y: jnp.where(update_target, x, y),
                        updated_dqn_state.network,
                        updated_dqn_state.target_network,
                    ))

                episode_rewards = episode_rewards.at[i].set(reward)
                episode_losses = episode_losses.at[i].set(jnp.sum(losses))
                return i+1, step_id+1, updated_dqn_state, weight_updated_rb_state, next_observation, done, epsilon, episode_rewards, episode_losses

            episode_length, step_id, dqn_state, rb_state, next_observation, done, epsilon, episode_rewards, episode_losses = jax.lax.while_loop(
                cond_fun,
                body_fun,
                (jnp.array(0), step_id, dqn_state, rb_state, environment_reset(), jnp.array(False), epsilon, jnp.empty((max_steps,)), jnp.empty((max_steps,)))
            )

            return step_id, episode_length, dqn_state, rb_state, epsilon, episode_rewards, episode_losses
        
        @jax.jit
        def evaluate_episode(network: scope.VariableDict, rb_state: fast_prioritised_rb.State) -> tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array]:

            def cond_fun(carry: tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]) -> jax.Array:
                i, rb_state, observation, done, episode_observations = carry
                return jnp.logical_and(i < evaluate_max_steps, jnp.logical_not(done))

            def body_fun(carry: tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]:
                i, rb_state, observation, done, episode_observations = carry
                discrete_action = get_greedy_action(network, observation)
                next_observation, distance_to_goal = environment_step(
                    observation,
                    jax_actions[discrete_action],
                )
                reward = - distance_to_goal
                done = distance_to_goal < 0.03
                updated_rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
                episode_observations = episode_observations.at[i+1].set(next_observation)
                return i+1, updated_rb_state, next_observation, done, episode_observations

            observation = environment_reset()
            episode_observations = jnp.empty((evaluate_max_steps+1, 2))
            episode_observations = episode_observations.at[0].set(observation)

            episode_length, updated_rb_state, next_observation, done, episode_observations = jax.lax.while_loop(
                cond_fun,
                body_fun,
                (jnp.array(0), rb_state, observation, jnp.array(False), episode_observations)
            )

            return episode_length, updated_rb_state, episode_observations, done

        @jax.jit
        def train_and_evaluate(step_id: jax.Array, dqn_state: jax_discrete_dqns.double_dqn.State, rb_state: fast_prioritised_rb.State, epsilon: jax.Array):
            step_id, episode_length, dqn_state, rb_state, epsilon, episode_rewards, episode_losses, = play_and_train(step_id, dqn_state, rb_state, epsilon)
            evaluate_episode_length, rb_state, evaluate_episode_observations, evaluate_has_reached_goal = evaluate_episode(dqn_state.network, rb_state)
            return step_id, dqn_state, rb_state, epsilon, episode_rewards, episode_losses, episode_length, evaluate_episode_observations, evaluate_episode_length, evaluate_has_reached_goal
        
        @jax.jit
        def train_and_evaluate_many(step_id: jax.Array, dqn_state: jax_discrete_dqns.double_dqn.State, rb_state: fast_prioritised_rb.State, epsilon: jax.Array):
            def body_fun(i: jax.Array, carry):
                step_id, dqn_state, rb_state, epsilon, episodes_rewards, episodes_losses, episodes_length, evaluate_episodes_observations, evaluate_episodes_length, evaluates_has_reached_goal = carry
                step_id, dqn_state, rb_state, epsilon, episode_rewards, episode_losses, episode_length, evaluate_episode_observations, evaluate_episode_length, evaluate_has_reached_goal = train_and_evaluate(step_id, dqn_state, rb_state, epsilon)
                episodes_rewards = episodes_rewards.at[i].set(episode_rewards)
                episodes_losses = episodes_losses.at[i].set(episode_losses)
                episodes_length = episodes_length.at[i].set(episode_length)
                evaluate_episodes_observations = evaluate_episodes_observations.at[i].set(evaluate_episode_observations)
                evaluate_episodes_length = evaluate_episodes_length.at[i].set(evaluate_episode_length)
                evaluates_has_reached_goal = evaluates_has_reached_goal.at[i].set(evaluate_has_reached_goal)
                return step_id, dqn_state, rb_state, epsilon, episodes_rewards, episodes_losses, episodes_length, evaluate_episodes_observations, evaluate_episodes_length, evaluates_has_reached_goal
            return jax.lax.fori_loop(0, max_episodes, body_fun, (
                step_id,
                dqn_state,
                rb_state,
                epsilon,
                jnp.empty((max_episodes, max_steps)),
                jnp.empty((max_episodes, max_steps)),
                jnp.empty((max_episodes,), dtype=jnp.int32),
                jnp.empty((max_episodes, evaluate_max_steps+1, 2)),
                jnp.empty((max_episodes,), dtype=jnp.int32),
                jnp.empty((max_episodes,), dtype=bool),
            ))
            
        # We pad the buffer with random transitions to ensure we don't trigger recompilation.
        rb_state = generate_random_rb()

        train_q_network = jax.jit(dqn.train_q_network)

        writer.add_scalar("epsilon", epsilon, 0)
        step_id, dqn_state, rb_state, epsilon, episodes_rewards, episodes_losses, episodes_length, evaluate_episodes_observations, evaluate_episodes_length, evaluates_has_reached_goal = train_and_evaluate_many(0, dqn_state, rb_state, epsilon)
        episodes_rewards = np.asarray(episodes_rewards)
        episodes_losses = np.asarray(episodes_losses)
        episodes_length = np.asarray(episodes_length)
        evaluate_episodes_observations = np.asarray(evaluate_episodes_observations)
        evaluates_has_reached_goal = np.asarray(evaluates_has_reached_goal)

        for episode_id in range(max_episodes):
            rewards = episodes_rewards[episode_id, :episodes_length[episode_id]]
            step_losses = episodes_losses[episode_id, :episodes_length[episode_id]]
            observations = evaluate_episodes_observations[episode_id, :evaluate_episodes_length[episode_id]+1]
            has_reached_goal = evaluates_has_reached_goal[episode_id].item()
            evaluate_reached_goal_count = np.sum(evaluates_has_reached_goal[:episode_id+1]).item()

            if len(rewards):
                log("reward", rewards, episode_id)
                writer.add_histogram("reward_dist", rewards, episode_id)
                writer.add_hparams(hyperparameters, metrics(rewards))

            if len(step_losses):
                log("loss", step_losses, episode_id)

            writer.add_scalar("reached_goal", has_reached_goal, episode_id)
            writer.add_scalar("reached_goal_count", evaluate_reached_goal_count, episode_id)
            rollout_tool.set_states(np.asarray(observations))
            log_greedy_policy()

        writer.add_scalar("epsilon", epsilon.item(), episode_id)

        get_q_values.update_q_network(dqn_state.network)
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
