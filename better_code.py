import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from continuous_agent import ContinuousAgent
from continuous_dqns import ContinuousDQNWithTargetNetwork, ContinuousDQN, \
    ContinuousDoubleDQN
from discrete_agent import DiscreteAgent
from discrete_dqns import DiscreteDQN, DiscreteDQNWithTargetNetwork
from environments import RandomEnvironment, BasicEnvironment
from replay_buffers import FastPrioritisedExperienceReplayBuffer
from tensorboard_writer import CustomSummaryWriter
from tools.actions_visual_tool import ActionsVisualTool
from tools.greedy_policy_graphics import GreedyPolicyTool
from datetime import datetime

if __name__ == "__main__":

    random_state = 816673
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    gamma = .9
    lr = 5e-4
    max_capacity = 10000
    batch_size = 64
    max_steps = 1000  # was 750
    max_episodes = 250
    epsilon = 1.
    delta = 0.0000071
    minimum_epsilon = 0.3
    sampling_eps = 1e-7
    weight_decay = 1e-7

    device = torch.device("cuda")
    display_game = False
    display_tools = False

    environment = RandomEnvironment(display=display_game, magnification=500)
    dqn = ContinuousDoubleDQN(gamma, lr, device=device, weight_decay=weight_decay)
    agent = ContinuousAgent(environment, dqn, stride=0.02)
    rb = FastPrioritisedExperienceReplayBuffer(max_capacity, batch_size,
                                               sampling_eps, agent)

    policy_tool = GreedyPolicyTool(magnification=250, agent=agent,
                                   max_step_num=200)
    # actions_tool = ActionsVisualTool(500, agent, 10)

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
    log_dir = os.path.join("runs", "continuous_better_code_runs", current_time)
    writer = CustomSummaryWriter(log_dir=log_dir)


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


    episodes_iter = tqdm(range(max_episodes))
    for episode_number in episodes_iter:
        episode_loss_list = []
        episode_reward_list = []
        agent.reset()
        if dqn.has_target_network():
            dqn.update_target_network()
        agent.dqn.train()
        for step_num in range(max_steps):
            transition, _ = agent.step(epsilon)
            state, action, reward, next_state = transition
            rb.store(state, action, reward, next_state)
            episode_reward_list.append(reward)

            if len(rb) > batch_size:
                transitions = rb.batch_sample().to(device)
                losses = dqn.train_q_network(transitions)
                rb.update_batch_weights(losses)
                episode_loss_list.append(losses.sum())

            if epsilon > minimum_epsilon:
                epsilon -= delta
                epsilon = max(epsilon, minimum_epsilon)
                episodes_iter.set_description(f"Epsilon: {epsilon:.3f}")

            # if display_tools and (step_num % 250 == 0):
            #     print("Drawing")
            #     policy_tool.draw()
            #     policy_tool.show()
            #     print("Finished Drawing")

        rewards = np.array(episode_reward_list)
        log("reward", rewards, episode_number)
        writer.add_histogram("reward_dist", rewards, episode_number)
        step_losses = np.array(episode_loss_list)
        log("loss", step_losses, episode_number)
        writer.add_hparams(hyperparameters, metrics(rewards))

        if display_tools:
            policy_tool.draw()
            log_greedy_policy(draw=False)
            policy_tool.show()
        else:
            log_greedy_policy()

    policy_tool.draw()
    policy_tool.save_image('greedy_policy_reward.png')
