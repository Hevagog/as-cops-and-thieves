from types import SimpleNamespace

TrainingConfig = SimpleNamespace(
    num_self_play_iterations=40,  # Total self-play iterations
    training_timesteps_per_role_training=100_000,  # Timesteps for training one role in one iteration
    archive_save_interval=1,  # Save to archive every N self-play iterations for each role
    policy_sample_strategy="pfsp",  # "latest", "random", or "pfsp"
    win_rate_buffer_size=20,  # For PFSP: number of recent games to consider for win rate
    n_trial_episodes=5,  # For evaluation in PFSP - selecting winner
    cop_role_prefix="cop",
    thief_role_prefix="thief",
)
