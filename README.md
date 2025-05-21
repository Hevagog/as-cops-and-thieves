# Agent Systems Cops and Thieves


[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

## Project Goal

This project aims to train intelligent, environment-agnostic agents for a "Cops and Thieves" scenario:

*   **Cops**: Learn to effectively search, patrol, chase, and apprehend thieves. With multiple cops, the goal is to observe emergent cooperative behaviors.
*   **Thieves**: Learn to strategically evade capture, hide, and navigate the environment to survive as long as possible.

The core challenge lies in enabling agents to analyze their surroundings and make adaptive decisions based on partial observations and past experiences, all within a dynamic, physics-based world. This project builds upon concepts from a previous endeavor, [*Chase model*](https://github.com/mzsuetam/model-poscigowy-sp), aiming for a more advanced and robust MARL-based solution.

## Key Features

*   **Multi-Agent Reinforcement Learning (MARL):** Utilizes MARL techniques for training multiple interacting agents.
*   **Fictitious Self-Play (FSP):** Agents are trained by playing against past versions of themselves and opponents, fostering an "arms race" of strategy development.
    *   **Policy-Space Response Oracles (PSRO) / Neural Fictitious Self-Play (NFSP) variants:** Employs advanced self-play mechanisms like Prioritized Fictitious Self-Play (PFSP) for opponent selection.
*   **Dynamic 2D Physics Environment:** Built with `pymunk` for realistic movement and interactions.
*   **Partial Observability:** Agents perceive the world through limited-range sensors (e.g., raycasting), mimicking real-world sensory limitations.
*   **Modular Design:** Separated components for agents, environments, models, and training utilities.

## Expected Agent Behaviors

| Agent Type | Expected Behaviors                                   |
| :--------- | :--------------------------------------------------- |
| **Cops**   | Strategic searching, coordinated chasing, efficient patrolling. |
| **Thieves**| Evasive maneuvers, intelligent hiding, risk assessment. |

## Technological Stack

This project leverages a modern stack for MARL research and development:

*   **[skrl](https://skrl.readthedocs.io/)**: A comprehensive and flexible library for Reinforcement Learning, used here for MARL algorithm implementation (MAPPO).
*   **[PettingZoo](https://pettingzoo.farama.org/)**: A standard API for multi-agent reinforcement learning environments, ensuring compatibility and best practices.
*   **[Pymunk](http://www.pymunk.org/)**: A robust 2D physics library used to create the simulation environment.
*   **[Pygame](https://www.pygame.org/)**: Used for visualizing the environment and agent interactions.
*   **PyTorch**: As the deep learning framework for neural network models.

## Getting Started

Sample usage:

```bash
python src/driver.py \  
    maps_templates/map-latest.json \    
    --map-image maps_templates/map-latest.png \
    --keep-alive
```

Available options:

```bash
python src/driver.py --help
```

## Custom SKRL Modifications for Phased Learning

To implement a phased learning approach where network components can be frozen for a specified duration at the beginning of a training run, custom modifications were introduced to the SKRL library and corresponding configurations were added.

### Custom Configuration Parameters

In `src/configs/mappo_config.py`, the `CFG_TRAINER` dictionary includes two custom parameters:

```python
CFG_TRAINER = {
    "timesteps": TrainingConfig.training_timesteps_per_role_training,
    # ... other parameters ...
    "opponent_freeze_duration": 20_000,  # How long to freeze opponent policy and value network NOTE: CUSTOM CFG
    "policy_freeze_duration": 8_000,  # How long to freeze the training agent's policy network NOTE: CUSTOM CFG
}
```

*   `opponent_freeze_duration`: Specifies the number of initial timesteps during which the opponent agent's value network (and potentially policy, depending on `agent_learning_utils.py` logic) will remain frozen. This is primarily used when loading an opponent from the policy archive.
*   `policy_freeze_duration`: Specifies the number of initial timesteps during which the policy network of the agent *currently being trained* will be frozen. This allows the value network to stabilize before policy updates begin.

### SKRL Library Modifications

To utilize these parameters, modifications were made to the SKRL trainer classes:

1.  **`skrl/trainers/torch/base.py`**:
    *   The `__init__` method of the `Trainer` class was updated to read and store these custom freeze duration parameters from the configuration.

    ```python
    # ...existing code...
        #------
        # added code
        if 'opponent_freeze_duration' in self.cfg and self.cfg['opponent_freeze_duration']:
            self.freeze_d = self.cfg['opponent_freeze_duration']
        else:
            self.freeze_d = 0
        if 'policy_freeze_duration' in self.cfg and self.cfg['policy_freeze_duration']:
            self.policy_freeze_d = self.cfg['policy_freeze_duration']
        else:
            self.policy_freeze_d = 0
        #------
    # ...existing code...
    ```
    *   The `multi_agent_train` method was modified to include logic for unfreezing networks based on these durations.

    ```python
    # ...existing code...
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            #----------------------------------
            # unfreeze opponent agent
            if self.freeze_d > 0 and timestep == self.freeze_d:
                for agent_name in self.agents.models:
                    self.agents.models[agent_name]["value"].freeze_parameters(False)
                tqdm.tqdm.write(f"Unfreezing opponent agent at timestep {timestep}")
            # unfreeze policy network
            if self.policy_freeze_d > 0 and timestep == self.policy_freeze_d:
                for agent_name in self.agents.models:
                    self.agents.models[agent_name]["policy"].freeze_parameters(False)
                tqdm.tqdm.write(f"Unfreezing policy network at timestep {timestep}")
            #----------------------------------
    # ...existing code...
    ```

2.  **`skrl/trainers/torch/sequential.py`**:
    *   Similar to `base.py`, the `train` method in `SequentialTrainer` (specifically the part handling `self.num_simultaneous_agents > 1`) was updated to unfreeze networks based on the stored `freeze_d` (for opponent) and `policy_freeze_d` (for training agent's policy) at the specified timesteps. 
    
    
    *Note: The provided `sequential.py` snippet shows unfreezing logic for `self.agents.models` which would apply to all agents managed by the `SequentialTrainer` if it were handling multiple distinct agent instances simultaneously. In the context of this project's self-play loop where one `MAPPO` agent instance (containing multiple policies/values for cops and thieves) is passed to the trainer, this unfreezing logic will apply to the networks within that single `MAPPO` instance.*

    ```python
    # ...existing code...
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            #----------------------------------
            # unfreeze opponent agent
            if self.freeze_d > 0 and timestep == self.freeze_d:
                for agent_name in self.agents.models:
                    # TODO: implement role-specific approach based on agent_learning_utils.py logic
                    # For now, it unfreezes all value networks.
                    # Consider how opponent_role_prefix would be passed or determined here if more granular control is needed directly in skrl.
                    self.agents.models[agent_name]["value"].freeze_parameters(False)
                tqdm.tqdm.write(f"Unfreezing opponent agent's value network at timestep {timestep}")
            # unfreeze policy network
            if self.policy_freeze_d > 0 and timestep == self.policy_freeze_d: # Corrected: check self.policy_freeze_d
                for agent_name in self.agents.models:
                    # TODO: implement role-specific approach based on agent_learning_utils.py logic
                    # It should only unfreeze the policy of the role_to_train_prefix
                    self.agents.models[agent_name]["policy"].freeze_parameters(False)
                tqdm.tqdm.write(f"Unfreezing training agent's policy network at timestep {timestep}")
            #----------------------------------
    # ...existing code...
    ```

**Important Note on SKRL Modifications:** The unfreezing logic within the SKRL files (`base.py` and `sequential.py`) currently unfreezes *all* policy or value networks within the `self.agents.models` dictionary when the respective `*_freeze_duration` timestep is reached. In your `agent_learning_utils.py`, you have more granular control, freezing specific roles (e.g., opponent role). The SKRL modification provides the *timing* for unfreezing. Ensure that the initial freezing state set in `agent_learning_utils.py` aligns with what you intend to be unfrozen by the SKRL trainer modifications. For instance, if an opponent's policy is meant to stay frozen for the entire training run of the other role, its initial `freeze_parameters(True)` call in `agent_learning_utils.py` should not be inadvertently overridden by the SKRL unfreezing logic if not desired. The current `policy_freeze_duration` correctly targets the training agent's policy, and `opponent_freeze_duration` targets the opponent's value network.
