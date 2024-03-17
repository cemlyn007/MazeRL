import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    run_id: str
    n_actions: int
    max_capacity: int
    mini_batches: int
    mini_batch_size: int
    max_steps: int
    evaluate_max_steps: int
    max_episodes: int
    epsilon: float
    delta: float
    minimum_epsilon: float
    sampling_eps: float
    tau: int
    gamma: float
    lr: float
    optimizer: str

def metrics(rewards) -> dict[str, float]:
    return {
        "metrics/mean_reward": np.mean(rewards).item(),
        "metrics/min_reward": np.min(rewards).item(),
        "metrics/max_reward": np.max(rewards).item(),
        "metrics/std_reward": np.std(rewards).item(),
        "metrics/median_reward": np.median(rewards).item(),
    }

def benchmark(config: Config) -> None:
    import warnings
    warnings.filterwarnings("error")
    import cv2
    import torch

    import helpers
    import tensorboard_writer
    import jax_discrete_dqns.double_dqn
    from environments import jax_random_environment
    import environments.renderer
    from jax_replay_buffers import fast_prioritised_rb
    from tools import episode_rollout_tool
    from tools.actions_visual_tool import ActionsVisualTool
    import random
    import jax
    import jax.numpy as jnp
    from flax.core import scope
    import optax

    class GetQValues:
        def __init__(self, q_network: jax_discrete_dqns.double_dqn.StatelessDiscreteDoubleDQN):
            self.q_network = q_network
            self._predict_q_values = jax.jit(self.q_network.predict_q_values)
            self._network_state = None

        def __call__(self, observations: np.ndarray) -> np.ndarray:
            return np.asarray(self._predict_q_values(self._network_state, jnp.asarray(observations)))
        
        def update_q_network(self, network: scope.VariableDict) -> None:
            self._network_state = network

    random_state = 816673
    torch.random.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    hps = helpers.JaxHyperparameters(
        gamma=config.gamma,
        lr=config.lr,
        mini_batches=config.mini_batches
    )

    environment = jax_random_environment.RandomEnvironment()
    renderer = environments.renderer.EnvironmentRenderer("RandomEnvironment", 500, np.asarray(environment.free_blocks), np.asarray(environment.goal_state))
    init_state = np.asarray(environment.reset())
    renderer.draw(init_state)
    if config.optimizer == "sgd":
        optimizer = optax.sgd(hps.lr)
    elif config.optimizer == "adam":
        optimizer = optax.adam(hps.lr)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    dqn = jax_discrete_dqns.double_dqn.StatelessDiscreteDoubleDQN(hps, config.n_actions, optimizer)

    def get_greedy_action(network: scope.VariableDict, observation: jax.Array) -> jax.Array:
        q_values = dqn.predict_q_values(network, jnp.expand_dims(observation, axis=0))
        return jnp.argmax(q_values)
    
    @jax.jit
    def select_action(network: scope.VariableDict, observation: jax.Array, key: jax.Array, epsilon: float) -> jax.Array:
        random_action_key, epsilon_key = jax.random.split(key)
        use_random = jax.random.uniform(epsilon_key) <= epsilon
        random_action = jax.random.randint(random_action_key, (), 0, config.n_actions)
        greedy_action = get_greedy_action(network, observation)
        return jnp.where(use_random, random_action, greedy_action)
    
    def create_actions(n_actions: int, stride: float) -> jax.Array:
        actions = []
        for i in range(n_actions):
            theta = 2. * np.pi * i / n_actions
            action = stride * np.array([np.cos(theta), np.sin(theta)])
            actions.append(helpers.trunc(action, 5))
        return jnp.stack(actions)
    
    jax_actions = create_actions(config.n_actions, 0.02)

    dqn_state = dqn.reset()
    
    rb = fast_prioritised_rb.StatelessFastPrioritisedExperienceReplayBuffer(
        config.max_capacity, config.mini_batches * config.mini_batch_size, config.sampling_eps, init_state.shape
    )
    rb_state = rb.reset()
    store = jax.jit(rb.store)
    batch_sample = jax.jit(rb.batch_sample)
    update_batch_weights = jax.jit(rb.update_batch_weights)

    rollout_tool = episode_rollout_tool.EpisodeRolloutTool(renderer.image)
    get_q_values = GetQValues(dqn)
    actions_tool = ActionsVisualTool(500, 15, config.n_actions, get_q_values)

    hyperparameters = {
        "random_state": random_state,
        "discrete_actions": True,
        "weighted_replay_buffer": True,
    }
    hyperparameters.update(dataclasses.asdict(config))

    writer = tensorboard_writer.CustomSummaryWriter(
        log_dir=f"runs/discrete_agent_runs/{config.run_id}"
    )

    def log(main_tag: str, values: np.ndarray, episode: int) -> None:
        writer.add_scalar(f"{main_tag}/mean", np.mean(values), episode)
        writer.add_scalar(f"{main_tag}/min", np.min(values), episode)
        writer.add_scalar(f"{main_tag}/max", np.max(values), episode)
        writer.add_scalar(f"{main_tag}/std", np.std(values), episode)
        writer.add_scalar(f"{main_tag}/median", np.median(values), episode)

    def log_greedy_policy(draw: bool=True) -> None:
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

    @jax.jit
    def generate_random_rb() -> fast_prioritised_rb.State:

        def body_fun(i: jax.Array, carry: tuple[fast_prioritised_rb.State, jax.Array]) -> tuple[fast_prioritised_rb.State, jax.Array]:
            rb_state, observation= carry
            discrete_action = jax.random.randint(jax.random.PRNGKey(i), (), 0, config.n_actions)
            next_observation, distance_to_goal = environment.step(
                observation,
                jax_actions[discrete_action],
            )
            reward = - distance_to_goal
            done = distance_to_goal < 0.03
            updated_rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
            next_observation = jnp.where(done, environment.reset(), next_observation)
            return updated_rb_state, next_observation

        updated_rb_state, _ = jax.lax.fori_loop(
            0,
            config.mini_batches * config.mini_batch_size,
            body_fun,
            (rb.reset(), environment.reset())
        )

        return updated_rb_state

    @jax.jit
    def play_and_train(step_id: jax.Array, dqn_state: jax_discrete_dqns.double_dqn.State, rb_state: fast_prioritised_rb.State, epsilon: jax.Array) -> tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]:

        def cond_fun(carry: tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
            i, step_id, dqn_state, rb_state, next_observation, done, epsilon, episode_rewards, episode_losses = carry
            return jnp.logical_and(i < config.max_steps, jnp.logical_not(done))

        def body_fun(carry: tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax_discrete_dqns.double_dqn.State, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            i, step_id, dqn_state, rb_state, observation, done, epsilon, episode_rewards, episode_losses = carry
            select_action_key, sample_key = jax.random.split(jax.random.PRNGKey(step_id))
            discrete_action = select_action(dqn_state.network, observation, select_action_key, epsilon)
            next_observation, distance_to_goal = environment.step(
                observation,
                jax_actions[discrete_action],
            )
            reward = - distance_to_goal
            done = distance_to_goal < 0.03
            updated_rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
            batch, indices = batch_sample(updated_rb_state, sample_key)
            updated_dqn_state, losses = dqn.train_q_network(dqn_state, batch)
            weight_updated_rb_state = update_batch_weights(updated_rb_state, indices, losses)

            epsilon = jnp.maximum(epsilon - config.delta, config.minimum_epsilon)
            if dqn.has_target_network:
                update_target = (step_id % config.tau) == 0
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
            (jnp.array(0), step_id, dqn_state, rb_state, environment.reset(), jnp.array(False), epsilon, jnp.empty((config.max_steps,)), jnp.empty((config.max_steps,)))
        )

        return step_id, episode_length, dqn_state, rb_state, epsilon, episode_rewards, episode_losses
    
    @jax.jit
    def evaluate_episode(network: scope.VariableDict, rb_state: fast_prioritised_rb.State) -> tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array]:

        def cond_fun(carry: tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]) -> jax.Array:
            i, rb_state, observation, done, episode_observations = carry
            return jnp.logical_and(i < config.evaluate_max_steps, jnp.logical_not(done))

        def body_fun(carry: tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, fast_prioritised_rb.State, jax.Array, jax.Array, jax.Array]:
            i, rb_state, observation, done, episode_observations = carry
            discrete_action = get_greedy_action(network, observation)
            next_observation, distance_to_goal = environment.step(
                observation,
                jax_actions[discrete_action],
            )
            reward = - distance_to_goal
            done = distance_to_goal < 0.03
            updated_rb_state = store(rb_state, observation, discrete_action, reward, done, next_observation)
            episode_observations = episode_observations.at[i+1].set(next_observation)
            return i+1, updated_rb_state, next_observation, done, episode_observations

        observation = environment.reset()
        episode_observations = jnp.empty((config.evaluate_max_steps+1, 2))
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
        return jax.lax.fori_loop(0, config.max_episodes, body_fun, (
            step_id,
            dqn_state,
            rb_state,
            epsilon,
            jnp.empty((config.max_episodes, config.max_steps)),
            jnp.empty((config.max_episodes, config.max_steps)),
            jnp.empty((config.max_episodes,), dtype=jnp.int32),
            jnp.empty((config.max_episodes, config.evaluate_max_steps+1, 2)),
            jnp.empty((config.max_episodes,), dtype=jnp.int32),
            jnp.empty((config.max_episodes,), dtype=bool),
        ))

    with jax.log_compiles(True):

        # We pad the buffer with random transitions to ensure we don't trigger recompilation.
        rb_state = generate_random_rb()

        writer.add_scalar("epsilon", config.epsilon, 0)
        step_id, dqn_state, rb_state, epsilon, episodes_rewards, episodes_losses, episodes_length, evaluate_episodes_observations, evaluate_episodes_length, evaluates_has_reached_goal = train_and_evaluate_many(0, dqn_state, rb_state, config.epsilon)
        episodes_rewards = np.asarray(episodes_rewards)
        episodes_losses = np.asarray(episodes_losses)
        episodes_length = np.asarray(episodes_length)
        evaluate_episodes_observations = np.asarray(evaluate_episodes_observations)
        evaluates_has_reached_goal = np.asarray(evaluates_has_reached_goal)

        for episode_id in range(config.max_episodes):
            rewards = episodes_rewards[episode_id, :episodes_length[episode_id]]
            step_losses = episodes_losses[episode_id, :episodes_length[episode_id]]
            observations = evaluate_episodes_observations[episode_id, :evaluate_episodes_length[episode_id]+1]
            has_reached_goal = evaluates_has_reached_goal[episode_id].item()
            evaluate_reached_goal_count = np.sum(evaluates_has_reached_goal[:episode_id+1]).item()

            if len(rewards):
                log("reward", rewards, episode_id)
                writer.add_histogram("reward_dist", rewards, episode_id)
                metrics_dict = metrics(rewards)
                metrics_dict["metrics/reached_goal"] = has_reached_goal
                metrics_dict["metrics/episode_length"] = evaluate_episodes_length[episode_id].item()
                writer.add_hparams(hyperparameters, metrics_dict)

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
        for mini_batches in [2, 4, 8, 16]:
            for mini_batch_size in [16, 32, 64]:
                for optimizer in ["sgd"]:
                    for tau in [50]:
                        for gamma in [0.95]:
                            for lr in [5e-4]:
                                config = Config(
                                    run_id=f"{sys.argv[1]}_tau_{tau}_gamma_{gamma}_lr_{lr}_{i}_optimizer_{optimizer}_mini_batch_size_{mini_batch_size}_mini_batches_{mini_batches}",
                                    n_actions=4,
                                    max_capacity=4096,
                                    mini_batches=mini_batches,
                                    mini_batch_size=mini_batch_size,
                                    max_steps=750,
                                    evaluate_max_steps=201,
                                    max_episodes=750,
                                    epsilon=1.0,
                                    delta=0.0000008,
                                    minimum_epsilon=0.1,
                                    sampling_eps=1e-7,
                                    tau=tau,
                                    gamma=gamma,
                                    lr=lr,
                                    optimizer=optimizer,
                                )
                                target = functools.partial(benchmark, config)
                                process = multiprocessing.Process(target=target)
                                start = time.monotonic()
                                process.start()
                                process.join()
                                end = time.monotonic()
                                print(f"Iteration {i} took {end - start:.2f} seconds")
