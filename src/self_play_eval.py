from environments import SimpleEnv
from maps import Map

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env


from configs import CFG_AGENT, CFG_AGENT_COP, CFG_AGENT_THIEF
from models import Policy, Value
from utils.model_utils import (
    copy_role_models,
    initialize_lstm_models_for_mappo,
    initialize_models_for_mappo,
)

if __name__ == "__main__":
    map = Map("maps_templates\\squarinth.json")
    env = SimpleEnv(map=map, render_mode="human", map_image=None, max_step_count=2000)
    env = wrap_env(env, wrapper="pettingzoo")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env.reset()  # Reset to get possible_agents

    # Create models - must match the architecture used during training
    models = initialize_lstm_models_for_mappo(
        observation_spaces=env.observation_space,
        action_spaces=env.action_space,
        possible_agents=env.possible_agents,
        device=device,
        base_obs_space_struct=env.get_base_observation_space_structure(),
    )

    # Initialize agent with the same configuration as during training
    cfg_cop = CFG_AGENT_COP.copy()
    cfg_cop["random_timesteps"] = 0  # No random actions during evaluation
    cfg_cop["learning_starts"] = 0  # No learning starts during evaluation
    cfg_thief = CFG_AGENT_THIEF.copy()
    cfg_thief["random_timesteps"] = 0  # No random actions during evaluation
    cfg_thief["learning_starts"] = 0  # No learning starts during evaluation
    # No need for memories during evaluation, but we need to pass an empty dict
    memories = {}
    cfg = CFG_AGENT.copy()
    cfg["random_timesteps"] = 0  # No random actions during evaluation
    cfg["learning_starts"] = 44440  # No learning starts during evaluation

    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=cfg.copy(),
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.get_nested_agent_observation_spaces(),
    )
    # thief_agent = MAPPO(
    #     possible_agents=env.possible_agents,
    #     models=models,
    #     memories=memories,  # Dummy memories
    #     cfg=cfg_thief,
    #     observation_spaces=env.observation_spaces,
    #     action_spaces=env.action_spaces,
    #     device=device,
    #     shared_observation_spaces=env.get_nested_agent_observation_spaces(),
    # )

    # Load checkpoint.
    # Note: This loads the entire agent state, including both the learning parameters
    # and the weights of all policy/value networks.
    cop_checkpoint_path = (
        "lstm_policy_archive_self_play_new\\joint_iter_19_full_agent.pt"
    )
    thief_checkpoint_path = (
        "lstm_policy_archive_self_play_new\\thieves\\thief_iter_1.pt"
    )
    agent.load(cop_checkpoint_path)
    agent.set_mode("eval")
    # thief_agent.load(cop_checkpoint_path)

    # copy_role_models(
    #     source_agent=cop_agent,
    #     target_agent=thief_agent,
    #     role_prefix="cop",
    #     possible_agents_list=env.possible_agents,
    #     device=device,
    # )

    cfg_trainer = {
        "timesteps": 20_000,  # Run for 1000 timesteps
        "headless": False,  # Show rendering
        "disable_progressbar": False,
        "close_environment_at_exit": True,
        "environment_info": "episode",
        "stochastic_evaluation": False,  # Use deterministic actions
    }

    # Run evaluation
    trainer = SequentialTrainer(env=env, cfg=cfg_trainer, agents=agent)
    trainer.eval()
