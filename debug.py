if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("error")
    import numpy as np

    import jax_discrete_agent
    import helpers
    import jax_discrete_dqns.double_dqn
    from environments import random_environment
    from jax_replay_buffers import fast_prioritised_rb
    import random
    import jax


    random_state = 816673
    random.seed(random_state)
    np.random.seed(random_state)

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

    device = jax.devices("cpu")[0]

    environment = random_environment.RandomEnvironment(
        display=False, magnification=500
    )

    dqn = jax_discrete_dqns.double_dqn.DiscreteDoubleDQN(hps, n_actions, device)
    agent = jax_discrete_agent.DiscreteAgent(environment, dqn, n_actions, stride=0.02)
    rb = fast_prioritised_rb.FastPrioritisedExperienceReplayBuffer(
        max_capacity, batch_size, sampling_eps, environment.init_state.shape
    )

    step_id = 0
    for episode_id in range(max_episodes):
        episode_loss_list = []
        episode_reward_list = []
        agent.reset()
        agent.dqn.train()
        for step_num in range(max_steps):
            transition, distance_to_goal = agent.step(0) # Debug so set to greedy!
            state, action, reward, next_state = transition
            done = distance_to_goal < 0.03
            rb.store(state, action, reward, done, next_state)
            episode_reward_list.append(reward)

            if len(rb) > batch_size:
                transitions = rb.batch_sample()
                losses = dqn.train_q_network(transitions)
                episode_loss_list.append(losses.sum())

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
            state, action, reward, next_state = transition
            done = distance_to_goal < 0.03
            states.append(agent.state)
            rb.store(state, action, reward, done, next_state)

            if done:
                evaluate_reached_goal_count += 1
                has_reached_goal = True
                break