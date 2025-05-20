import torch
from pathlib import Path
import os

from environments import SimpleEnv
from maps import Map
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory  # Required for MAPPO initialization

from models import Policy, Value
from utils.policy_archive_utils import (
    add_policy_to_archive,
    sample_policy_from_archive,
    get_latest_policy_from_archive,
    update_policy_win_rate,
)
from utils.model_utils import copy_role_models, initialize_models_for_mappo
from utils.eval_pfsp_agents import evaluate_agents

# Configuration
NUM_SELF_PLAY_ITERATIONS = 40  # Total self-play iterations
TRAINING_TIMESTEPS_PER_ROLE_TRAINING = (
    20_000  # Timesteps for training one role in one iteration
)
ARCHIVE_SAVE_INTERVAL = 1  # Save to archive every N self-play iterations for each role
POLICY_SAMPLE_STRATEGY = "pfsp"  # "latest", "random", or "pfsp"
WIN_RATE_BUFFER_SIZE = 20  # For PFSP: number of recent games to consider for win rate
N_TRIAL_EPISODES = 5  # For evaluation in PFSP - selecting winner

COP_ROLE_PREFIX = "cop"
THIEF_ROLE_PREFIX = "thief"


def train_role(
    env,
    training_agent: MAPPO,
    role_to_train_prefix: str,
    opponent_role_prefix: str,
    current_role_checkpoint_path: str | None,
    opponent_role_archive_path: Path,
    cfg_trainer: dict,
    device: torch.device,
):
    """
    Manages a training phase for a specific role.
    1. Loads the latest policy for the role being trained (if exists).
    2. Loads a sampled policy (e.g., via PFSP) for the opponent role from its archive.
    3. Freezes the opponent's policy.
    4. Trains the agent.
    5. Updates win-rate for the opponent policy based on evaluation during training.
    6. Returns the trained agent.
    """
    print(f"--- Training {role_to_train_prefix}s ---")

    # Initialize or load state for the training_agent
    # The training_agent's models are re-initialized fresh or loaded before this function usually.
    # However, we need to ensure the role_to_train starts from its latest checkpoint,
    # and opponents are from their archive.

    if current_role_checkpoint_path:
        print(
            f"Loading {role_to_train_prefix} state from: {current_role_checkpoint_path}"
        )
        training_agent.load(
            current_role_checkpoint_path
        )  # Loads all models in the agent

    # Sample and load opponent policy
    opponent_checkpoint_path = sample_policy_from_archive(
        opponent_role_archive_path,
        opponent_role_prefix,
        strategy=POLICY_SAMPLE_STRATEGY,
    )
    opponent_policy_filename = None
    if opponent_checkpoint_path:
        opponent_policy_filename = Path(opponent_checkpoint_path).name
        print(
            f"Loading opponent ({opponent_role_prefix}) policy from: {opponent_checkpoint_path}"
        )
        # Create a temporary agent to load the opponent's full state
        temp_opponent_loader_agent_models = initialize_models_for_mappo(
            env.observation_space,
            env.action_space,
            env.possible_agents,
            device,
            env.get_base_observation_space_structure(),
        )
        temp_memories = {
            name: RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)
            for name in env.possible_agents
        }

        temp_opponent_loader_agent = MAPPO(
            possible_agents=env.possible_agents,
            models=temp_opponent_loader_agent_models,
            memories=temp_memories,
            cfg=MAPPO_DEFAULT_CONFIG.copy(),
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device,
            shared_observation_spaces=env.get_nested_agent_observation_spaces(),
        )
        temp_opponent_loader_agent.load(opponent_checkpoint_path)
        copy_role_models(
            temp_opponent_loader_agent,
            training_agent,
            opponent_role_prefix,
            env.possible_agents,
            device,
        )
        del temp_opponent_loader_agent  # Free memory
        # torch.cuda.empty_cache() if device.type == "cuda" else None

        # --- Freeze opponent policies for Fictitious Play ---
        print(f"Freezing policies for opponent role: {opponent_role_prefix}")
        for agent_name in training_agent.models:
            if agent_name.startswith(opponent_role_prefix):
                training_agent.models[agent_name]["policy"].freeze_parameters()
                training_agent.models[agent_name]["value"].freeze_parameters()
            elif agent_name.startswith(
                role_to_train_prefix
            ):  # Ensure learning role is trainable
                training_agent.models[agent_name]["policy"].freeze_parameters(False)
                training_agent.models[agent_name]["value"].freeze_parameters(False)
    else:
        print(
            f"No opponent policy found in archive for {opponent_role_prefix}. Opponents will use their current/initial policies in training_agent."
        )

    # Configure and run trainer
    trainer = SequentialTrainer(env=env, cfg=cfg_trainer, agents=training_agent)
    trainer.train()

    # Trainer automatically calls agent.save() if checkpointing is enabled in its config.
    # For self-play, we manage saving explicitly after each role's training.
    # The path used by trainer's internal checkpointing might not be what we want for archive.

    # --- Update win-rate for the opponent policy that was used (if any) ---
    if opponent_checkpoint_path:
        avg_cop_return, avg_thief_return = evaluate_agents(
            env, training_agent, N_TRIAL_EPISODES
        )
        print(f"Avg Cop Return: {avg_cop_return}, Avg Thief Return: {avg_thief_return}")
        if role_to_train_prefix == COP_ROLE_PREFIX:
            opponent_won = avg_thief_return > avg_cop_return
        elif role_to_train_prefix == THIEF_ROLE_PREFIX:
            opponent_won = avg_cop_return > avg_thief_return

        update_policy_win_rate(
            opponent_role_archive_path,
            opponent_policy_filename,
            opponent_won,
            WIN_RATE_BUFFER_SIZE,
        )

    # We need to return a path to the *newly trained* policy for this role.
    # The training_agent now contains the updated policies for role_to_train
    # AND policies for opponent_role that it trained against.

    return training_agent  # Return the agent, saving will be handled outside


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA errors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)  # Context-manager
    print(f"Using device: {device}")

    # Environment setup
    map_file = Path("maps_templates/lbirinth.json")  # Or use argparse
    game_map = Map(map_file)
    env = SimpleEnv(map=game_map, render_mode="rgb_array")
    env = wrap_env(env, wrapper="pettingzoo")

    # Policy Archive Setup
    base_archive_path = Path("policy_archive_self_play")
    cop_archive_path = base_archive_path / "cops"
    thief_archive_path = base_archive_path / "thieves"
    cop_archive_path.mkdir(parents=True, exist_ok=True)
    thief_archive_path.mkdir(parents=True, exist_ok=True)

    # Paths to the *current best* or *latest fully trained* policies for each role
    # These are distinct from the archive, which stores historical versions.
    # These paths will point to full agent checkpoints.
    current_cop_checkpoint = get_latest_policy_from_archive(
        cop_archive_path, COP_ROLE_PREFIX
    )
    current_thief_checkpoint = get_latest_policy_from_archive(
        thief_archive_path, THIEF_ROLE_PREFIX
    )

    cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
    cfg_agent.update(
        {
            "rollouts": 4096,
            "learning_epochs": 10,
            "mini_batches": 16,  # Adjusted from 8 to 16
            "entropy_loss_scale": 0.01,  # Adjusted from 0.05
            "random_timesteps": 1500,  # Adjusted from 5000
            "learning_starts": 4000,  # Adjusted from 8000
            "learning_rate": 3e-4,  # Adjusted from 1e-4
            "ratio_clip": 0.2,  # Adjusted from 0.1
            "kl_threshold": 0.05,  # Adjusted from 0.01
            "enable_kl": True,  # Was False, now True
            "value_loss_scale": 0.5,  # Default is 1.0, common to use 0.5
            "grad_norm_clip": 0.5,  # Default is 0.5, common practice
        }
    )

    cfg_trainer = {
        "timesteps": TRAINING_TIMESTEPS_PER_ROLE_TRAINING,
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": False,
        "checkpoint_interval": -1,
        "evaluation_interval": TRAINING_TIMESTEPS_PER_ROLE_TRAINING
        // 5,  # Evaluate a few times per role training
        "evaluation_episodes": 5,  # Number of episodes for each evaluation run
        "opponent_freeze_duration": 5_000,  # How long to freeze opponent policy NOTE: CUSTOM CFG
    }

    # Initialize one MAPPO agent instance that will be reconfigured/reloaded.
    # Models will be initialized fresh or loaded inside the loop.
    # SKRL memories for MAPPO
    env.reset()
    memories = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(
            memory_size=2048,
            device=device,
        )

    start_iteration = 10
    # Main Self-Play Loop
    for iteration in range(start_iteration, NUM_SELF_PLAY_ITERATIONS + start_iteration):
        print(
            f"\n===== Self-Play Iteration: {iteration + 1}/{NUM_SELF_PLAY_ITERATIONS} ====="
        )

        # --- Train Cops ---
        # Initialize/Re-initialize models for the agent for this training phase
        # This ensures that optimizers etc. are fresh or correctly reloaded.
        models_cop_phase = initialize_models_for_mappo(
            env.observation_space,
            env.action_space,
            env.possible_agents,
            device,
            env.get_base_observation_space_structure(),
        )
        mappo_agent_cops = MAPPO(
            possible_agents=env.possible_agents,
            models=models_cop_phase,
            memories=memories,
            cfg=cfg_agent,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device,
            shared_observation_spaces=env.get_nested_agent_observation_spaces(),
        )

        mappo_agent_cops = train_role(
            env,
            mappo_agent_cops,
            COP_ROLE_PREFIX,
            THIEF_ROLE_PREFIX,
            current_cop_checkpoint,
            thief_archive_path,
            cfg_trainer,
            device,
        )
        # Save the state of the agent after cop training. This checkpoint contains newly trained cops
        # and the (also updated) thieves they trained against.
        new_cop_checkpoint_path_str = str(
            base_archive_path / f"cop_iter_{iteration}_full_agent.pt"
        )
        mappo_agent_cops.save(new_cop_checkpoint_path_str)
        current_cop_checkpoint = new_cop_checkpoint_path_str
        if iteration % ARCHIVE_SAVE_INTERVAL == 0 or iteration == (
            start_iteration + NUM_SELF_PLAY_ITERATIONS - 1
        ):
            add_policy_to_archive(
                current_cop_checkpoint, cop_archive_path, iteration, COP_ROLE_PREFIX
            )
        del mappo_agent_cops  # Clean up
        # torch.cuda.empty_cache() if device.type == "cuda" else None

        # --- Train Thieves ---
        # Re-initialize models for the agent for this training phase
        models_thief_phase = initialize_models_for_mappo(
            env.observation_space,
            env.action_space,
            env.possible_agents,
            device,
            env.get_base_observation_space_structure(),
        )
        mappo_agent_thieves = MAPPO(
            possible_agents=env.possible_agents,
            models=models_thief_phase,
            memories=memories,
            cfg=cfg_agent,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device,
            shared_observation_spaces=env.get_nested_agent_observation_spaces(),
        )

        mappo_agent_thieves = train_role(
            env,
            mappo_agent_thieves,
            THIEF_ROLE_PREFIX,
            COP_ROLE_PREFIX,
            current_thief_checkpoint,
            cop_archive_path,
            cfg_trainer,
            device,
        )

        new_thief_checkpoint_path_str = str(
            base_archive_path / f"thief_iter_{iteration}_full_agent.pt"
        )
        mappo_agent_thieves.save(new_thief_checkpoint_path_str)
        current_thief_checkpoint = new_thief_checkpoint_path_str
        if iteration % ARCHIVE_SAVE_INTERVAL == 0 or iteration == (
            start_iteration + NUM_SELF_PLAY_ITERATIONS - 1
        ):
            add_policy_to_archive(
                current_thief_checkpoint,
                thief_archive_path,
                iteration,
                THIEF_ROLE_PREFIX,
            )
        del mappo_agent_thieves  # Clean up
        # torch.cuda.empty_cache() if device.type == "cuda" else None
        # Clean up memory if needed, e.g. del mappo_agent if it's large and re-instantiated fully
        # torch.cuda.empty_cache() # If memory issues persist

    env.close()
