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