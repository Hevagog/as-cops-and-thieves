
# 2nd Progress Update

## Change Log

### Environment

Since the last report, we have undertaken significant refactoring and bug-fixing efforts to ensure compliance with the skrl framework.

### skrl & RL Algorithm

For our Multi-Agent Reinforcement Learning (MARL) task, we have chosen to utilize MAPPO (Multi-Agent Proximal Policy Optimization) due to the cooperative nature of our environment.  
This phase involved:

- Introducing new spaces: *shared observation space* and *state space*
- Restructuring the observation and action spaces to fully leverage PyTorch tensors

---

### Agents

On the agent side, we implemented minor fixes and further optimizations in the vision controller.  
We are also addressing ongoing challenges with the reward function, where agents are exploiting the reward mechanism, leading to unintended behaviors during episodes.

## Models

For the MAPPO implementation, we developed both policy and value networks.  
Both networks use a Multi-Layer Perceptron (MLP) architecture with two hidden layers.  
The policy network outputs the best possible action for a given state, while the value network serves as a critic, estimating the expected return for the agent's actions.

---

### Policy Network

Utilizes `CategoricalMixin` base skrl class due to its stochastic nature, and because of the categorical action space.

![Concept of policy network. [Source](https://skrl.readthedocs.io/en/latest/api/models/categorical.html)](img/model_categorical-dark.svg){width=80%}

---

### Value Network
  
Utilizes `DeterministicMixin` base skrl class. Takes the global state of the environment.

![Concept of value network. [Source](https://skrl.readthedocs.io/en/latest/api/models/deterministic.html)](img/model_deterministic-dark.svg){width=70%}
