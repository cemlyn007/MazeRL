# MazeRL

Using Deep Q-Learning, the objective here is to train an agent to navigate through a maze. In this implementation, I
have created agents that can handle discrete or continuous actions.

| ![Alt text](images/discrete_greedy_policy.png?raw=true 'Discrete Greedy Policy')  | ![Alt text](images/continuous_greedy_policy.png?raw=true 'Continuous Greedy Policy') |
|:---:|:---:|

## Currently

Possible models the agent can use:

* Deep Q-Learning
* Deep Q-Learning with a Target Network
* Double Deep Q-Learning

Actions:

* For discrete actions, the agent can only consider up, down, left right. Each action moves by a fixed stride.
* For continuous actions, the agent can only consider what angle it wants to take. The mean angle is sampled using the
  cross entropy method.

Tools:

* Greedy Policy Tool that allows the user to visualise the current greedy policy that the agent has learnt. Images of
  the tool can be seen above.
* Action Visual Tool which discretises the environment and shows the agents preferred order of discrete actions at any
  grid point. The actions the agent would take are strong yellow, with strong blue being the least preferred action that
  the agent would consider taking.

![Alt text](images/actions_visualisation.png?raw=true 'Action Visualisation')

## Going Further

In the future, I would like to introduce:

* For discrete actions, be able to consider N directions.
* Policy-based methods.

