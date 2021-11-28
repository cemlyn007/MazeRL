import os
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

import continuous_agent
import continuous_dqns.double_dqn
import environments.random_environment
import tensorboard_writer
import tools.greedy_policy_graphics
from replay_buffers import fast_prioritised_rb

if __name__ == "__main__":

    random_state = 816673
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    gamma = .9
    lr = 0.0003
    max_capacity = 10000
    batch_size = 256 * 3
    max_steps = 1000  # was 750
    max_episodes = 50  # was 250
    epsilon = 1.
    delta = 0.000071
    minimum_epsilon = 0.3
    sampling_eps = 1e-7
    weight_decay = 1e-7
    tau = 5  # target network episode update rate

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    display_game = False
    display_tools = False

    environment = environments.random_environment.RandomEnvironment(display=display_game,
                                                                    magnification=500)
    dqn = continuous_dqns.double_dqn.ContinuousDoubleDQN(gamma, lr, device=device,
                                                         weight_decay=weight_decay)
    agent = continuous_agent.ContinuousAgent(environment, dqn, stride=0.02)
    rb = fast_prioritised_rb.FastPrioritisedExperienceReplayBuffer(max_capacity, batch_size,
                                                                   sampling_eps, agent)

    policy_tool = tools.greedy_policy_graphics.GreedyPolicyTool(magnification=250, agent=agent,
                                                                max_step_num=200)

    hyperparameters = {
        "gamma": gamma,
        "lr": lr,
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
        return {"metrics/mean_reward": np.mean(rewards),
                "metrics/min_reward": np.min(rewards),
                "metrics/max_reward": np.max(rewards),
                "metrics/std_reward": np.std(rewards),
                "metrics/median_reward": np.median(rewards)}


    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join("runs", "continuous_actions_runs", current_time)
    writer = tensorboard_writer.CustomSummaryWriter(log_dir=log_dir)


    def log(main_tag, values, episode):
        writer.add_scalar(f"{main_tag}/mean", np.mean(values), episode)
        writer.add_scalar(f"{main_tag}/min", np.min(values), episode)
        writer.add_scalar(f"{main_tag}/max", np.max(values), episode)
        writer.add_scalar(f"{main_tag}/std", np.std(values), episode)
        writer.add_scalar(f"{main_tag}/median", np.median(values), episode)


    def log_greedy_policy(draw=True):
        if draw:
            policy_tool.draw()
        policy_img = cv2.cvtColor(policy_tool.image, cv2.COLOR_BGR2RGB)
        policy_img = torch.from_numpy(policy_img)
        writer.add_image("greedy_policy", policy_img, episode_number,
                         dataformats="HWC")


    model_path = os.path.join("models", "continuous_models")
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    episodes_iter = tqdm(range(max_episodes))
    for episode_number in episodes_iter:
        has_reached_goal = False
        episode_loss_list = []
        episode_reward_list = []
        agent.reset()
        if dqn.has_target_network() and (episode_number % tau == 0):
            dqn.update_target_network()
        agent.dqn.train()
        for step_num in range(max_steps):
            transition, distance_to_goal = agent.step(epsilon)
            state, action, reward, next_state = transition
            rb.store(state, action, reward, next_state)
            episode_reward_list.append(reward)

            if len(rb) > rb.batch_size:
                transitions = rb.batch_sample().to(device)
                losses = dqn.train_q_network(transitions)
                rb.update_batch_weights(losses)
                episode_loss_list.append(losses.sum())

            if epsilon > minimum_epsilon:
                epsilon -= delta
                epsilon = max(epsilon, minimum_epsilon)
                episodes_iter.set_description(f"Epsilon: {epsilon:.3f} ")

        agent.dqn.eval()
        agent.reset()
        for step_num in range(max_steps):
            transition, distance_to_goal = agent.step(0.0)
            state, action, reward, next_state = transition
            rb.store(state, action, reward, next_state)

            if distance_to_goal < 0.03:
                has_reached_goal = True
                break

        rewards = np.array(episode_reward_list)
        log("reward", rewards, episode_number)
        writer.add_histogram("reward_dist", rewards, episode_number)
        step_losses = np.array(episode_loss_list)
        log("loss", step_losses, episode_number)
        writer.add_hparams(hyperparameters, metrics(rewards))
        writer.add_scalar("reached_goal", has_reached_goal, episode_number)
        writer.add_scalar("epsilon", epsilon, episode_number)

        if display_tools:
            policy_tool.draw()
            log_greedy_policy(draw=False)
            policy_tool.show()
        else:
            log_greedy_policy()

        torch.save(dqn.q_network.state_dict(),
                   os.path.join(model_path, f"q_networks_state_dict-{episode_number}.pt"))
        torch.save(dqn.target_network.state_dict(),
                   os.path.join(model_path, f"target_networks_state_dict-{episode_number}.pt"))

    policy_tool.draw()
    policy_tool.save_image('greedy_policy_reward.png')

    torch.save(dqn.q_network.state_dict(),
               os.path.join(model_path, "q_networks_state_dict.pt"))
    torch.save(dqn.target_network.state_dict(),
               os.path.join(model_path, "target_networks_state_dict.pt"))
