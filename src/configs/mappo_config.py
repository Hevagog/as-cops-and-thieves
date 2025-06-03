from skrl.multi_agents.torch.mappo import MAPPO_DEFAULT_CONFIG
from configs.training_config import TrainingConfig

# Base configuration shared by both
_BASE_AGENT_CONFIG = MAPPO_DEFAULT_CONFIG.copy()
_BASE_AGENT_CONFIG.update(
    {
        "rollouts": 4096,
        "random_timesteps": 10000,
        "learning_starts": 15000,
        "kl_threshold": 0.015,
        "enable_kl": True,
        "value_loss_scale": 0.5,
        "grad_norm_clip": 0.5,
    }
)


CFG_AGENT_COP = _BASE_AGENT_CONFIG.copy()
CFG_AGENT_COP.update(
    {
        "learning_epochs": 4,
        "mini_batches": 4,  # Results in batch_size 1024 for rollouts 4096
        "entropy_loss_scale": 0.02,
        "learning_rate": 1e-4,
        "ratio_clip": 0.15,
    }
)

CFG_AGENT_THIEF = _BASE_AGENT_CONFIG.copy()
CFG_AGENT_THIEF.update(
    {
        "learning_epochs": 3,
        "mini_batches": 8,  # Results in batch_size 512 for rollouts 4096
        "entropy_loss_scale": 0.01,
        "learning_rate": 3e-4,  # Or 2e-4 if cops still struggle significantly
        "ratio_clip": 0.2,
    }
)

CFG_AGENT = _BASE_AGENT_CONFIG.copy()
CFG_AGENT.update(
    {
        "learning_epochs": 4,
        "mini_batches": 4,  # Results in batch_size 1024 for rollouts 4096
        "entropy_loss_scale": 0.02,
        "learning_rate": 1e-4,
        "ratio_clip": 0.15,
    }
)

CFG_TRAINER = {
    "timesteps": TrainingConfig.training_timesteps_per_role_training,
    "headless": True,
    "disable_progressbar": False,
    "close_environment_at_exit": False,
    "checkpoint_interval": -1,
    "evaluation_interval": TrainingConfig.training_timesteps_per_role_training
    // 10,  # Evaluate a few times per role training
    "evaluation_episodes": 5,  # Number of episodes for each evaluation run
    "opponent_freeze_duration": 15_000,  # How long to freeze opponent policy and value network NOTE: CUSTOM CFG - see README.md
    "policy_freeze_duration": 15_000,  # How long to freeze the training agent's policy network NOTE: CUSTOM CFG - see README.md
}
