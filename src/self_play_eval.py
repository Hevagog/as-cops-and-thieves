from environments import SimpleEnv
from maps import Map

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch

from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env

from models import Policy, Value
from utils.model_utils import copy_role_models

if __name__ == "__main__":
    map = Map("maps_templates\\lbirinth.json")
    env = SimpleEnv(map=map, render_mode="human", map_image=None)
    env = wrap_env(env, wrapper="pettingzoo")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models - must match the architecture used during training
    models = {}
    env.reset()  # Reset to get possible_agents
    for agent_name in env.possible_agents:
        models[agent_name] = {}
        models[agent_name]["policy"] = Policy(
            env.observation_space(agent_name),
            env.action_space(agent_name),
            device,
        )

        models[agent_name]["value"] = Value(
            env.get_base_observation_space_structure(),
            env.action_space(agent_name),
            device,
            agent_count=len(env.possible_agents),
        )

    # Initialize agent with the same configuration as during training
    cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
    cfg_agent.update(
        {
            "rollouts": 4096,
            "learning_epochs": 8,
            "mini_batches": 16,
            "entropy_loss_scale": 0.05,  # Encourage more exploration
            "random_timesteps": 10000,  # More random exploration at start
            "learning_starts": 20000,  # Start learning after more exploration
            "learning_rate": 1e-4,  # Lower for stability
            "ratio_clip": 0.1,  # More conservative updates
            "kl_threshold": 0.01,  # Early stop if policy changes too much
            "enable_kl": True,
        }
    )

    # No need for memories during evaluation, but we need to pass an empty dict
    memories = {}

    cop_agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,  # Dummy memories
        cfg=MAPPO_DEFAULT_CONFIG.copy(),  # Basic config
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.get_nested_agent_observation_spaces(),
    )
    thief_agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,  # Dummy memories
        cfg=MAPPO_DEFAULT_CONFIG.copy(),  # Basic config
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.get_nested_agent_observation_spaces(),
    )

    # Load checkpoint.
    # Note: This loads the entire agent state, including both the learning parameters
    # and the weights of all policy/value networks.
    cop_checkpoint_path = "policy_archive_self_play\\cop_iter_4_full_agent.pt"
    thief_checkpoint_path = "policy_archive_self_play\\thief_iter_2_full_agent.pt"
    cop_agent.load(cop_checkpoint_path)
    thief_agent.load(thief_checkpoint_path)

    copy_role_models(
        source_agent=cop_agent,
        target_agent=thief_agent,
        role_prefix="cop",
        possible_agents_list=env.possible_agents,
        device=device,
    )

    cfg_trainer = {
        "timesteps": 1800,  # Run for 1000 timesteps
        "headless": False,  # Show rendering
        "disable_progressbar": False,
        "close_environment_at_exit": True,
        "environment_info": "episode",
        "stochastic_evaluation": False,  # Use deterministic actions
    }

    # Run evaluation
    trainer = SequentialTrainer(env=env, cfg=cfg_trainer, agents=thief_agent)

    trainer.eval()
