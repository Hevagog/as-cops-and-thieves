from skrl.multi_agents.torch.mappo import MAPPO_DEFAULT_CONFIG
from configs.training_config import TrainingConfig

CFG_AGENT = MAPPO_DEFAULT_CONFIG.copy()
CFG_AGENT.update(
    {
        "rollouts": 4096,
        "learning_epochs": 3,
        "mini_batches": 16,  # Adjusted from 8 to 16
        "entropy_loss_scale": 0.01,  # Adjusted from 0.05
        "random_timesteps": 1500,  # Adjusted from 5000
        "learning_starts": 2000,  # Adjusted from 8000
        "learning_rate": 3e-4,  # Adjusted from 1e-4
        "ratio_clip": 0.2,  # Adjusted from 0.1
        "kl_threshold": 0.05,  # Adjusted from 0.01
        "enable_kl": True,  # Was False, now True
        "value_loss_scale": 0.5,  # Default is 1.0, common to use 0.5
        "grad_norm_clip": 0.5,  # Default is 0.5, common practice
    }
)

CFG_TRAINER = {
    "timesteps": TrainingConfig.training_timesteps_per_role_training,
    "headless": True,
    "disable_progressbar": False,
    "close_environment_at_exit": False,
    "checkpoint_interval": -1,
    "evaluation_interval": TrainingConfig.training_timesteps_per_role_training
    // 5,  # Evaluate a few times per role training
    "evaluation_episodes": 5,  # Number of episodes for each evaluation run
    "opponent_freeze_duration": 20_000,  # How long to freeze opponent policy and value network NOTE: CUSTOM CFG - see README.md
    "policy_freeze_duration": 8_000,  # How long to freeze the training agent's policy network NOTE: CUSTOM CFG - see README.md
}
