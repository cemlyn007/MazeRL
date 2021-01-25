class AbstractAgent:

    def __init__(self, environment, dqn):
        self.environment = environment
        self.dqn = dqn
        self.state = None
        self.total_reward = None
        self.reset()

    def reset(self):
        self.state = self.environment.reset()
        self.total_reward = 0.0

    def step(self, epsilon=0):
        raise NotImplementedError

    @staticmethod
    def compute_reward(distance_to_goal):
        reward = (1 - distance_to_goal).astype(distance_to_goal)
        return reward

    def get_greedy_action(self, state):
        raise NotImplementedError