import cv2
import numpy as np
import torch
from tqdm import tqdm

from tools.actions_visual_tool import ActionsVisualTool
from agents import Agent
from dqns.double_dqn import DoubleDQN
from tools.greedy_policy_graphics import GreedyPolicyTool
from environments.random_environment import RandomEnvironment
from datetime import datetime

from replay_buffers import SlowPrioritisedExperienceReplayBuffer

from tensorboard_writer import CustomSummaryWriter

if __name__ == "__main__":

    random_state = 816673
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    gamma = .9
    lr = 1e-5
    max_capacity = 10000
    batch_size = 256  # 50
    max_steps = 750
    max_episodes = 250
    epsilon = 1.
    delta = 0.0000071
    minimum_epsilon = 0.3
    sampling_eps = 1e-8
    alpha = 2.

    device = torch.device("cuda")
    display_game = False
    display_tools = False

    environment = RandomEnvironment(display=display_game, magnification=500)
    dqn = DoubleDQN(gamma, lr, device=device)
    agent = Agent(environment, dqn, stride=0.02)
    # rb = ReplayBuffer(max_capacity, batch_size)
    rb = SlowPrioritisedExperienceReplayBuffer(max_capacity, batch_size,
                                               sampling_eps, alpha, agent)

    policy_tool = GreedyPolicyTool(magnification=250, agent=agent,
                                   max_step_num=200)
    actions_tool = ActionsVisualTool(500, agent, 10)

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
        "alpha": alpha,
    }


    def metrics(rewards):
        return {"metrics/mean_reward": np.mean(rewards),
                "metrics/min_reward": np.min(rewards),
                "metrics/max_reward": np.max(rewards),
                "metrics/std_reward": np.std(rewards),
                "metrics/median_reward": np.median(rewards)}


    now = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = CustomSummaryWriter(log_dir=f"runs/starter_code_runs/{now}")


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
                loss = dqn.train_q_network(transitions)
                episode_loss_list.append(loss)

            if epsilon > minimum_epsilon:
                epsilon -= delta
                epsilon = max(epsilon, minimum_epsilon)
                episodes_iter.set_description(f"Epsilon: {epsilon:.3f}")

        rewards = np.array(episode_reward_list)
        log("reward", rewards, episode_number)
        writer.add_histogram("reward_dist", rewards, episode_number)
        losses = np.array(episode_loss_list)
        log("loss", losses, episode_number)
        writer.add_hparams(hyperparameters, metrics(rewards))

        if display_tools:
            actions_tool.draw()
            policy_tool.draw()
            log_greedy_policy(draw=False)
            actions_tool.show()
            policy_tool.show()
        else:
            log_greedy_policy()

    # policy_tool.draw()
    # policy_tool.save_image('greedy_policy_reward.png')
