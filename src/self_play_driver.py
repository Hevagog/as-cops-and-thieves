import torch
from pathlib import Path
import os
from copy import copy
from types import SimpleNamespace

from environments import SimpleEnv
from maps import Map
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

from utils.policy_archive_utils import (
    add_policy_to_archive,
    get_latest_policy_from_archive,
)
from utils.model_utils import initialize_models_for_mappo
from utils.agent_learning_utils import train_role
from configs import CFG_AGENT, CFG_TRAINER, TrainingConfig


# Configuration
tc = copy(TrainingConfig)


def _orchestrate_training_phase(
    env,
    role_to_train_prefix: str,
    opponent_role_prefix: str,
    current_role_checkpoint_path: str | None,
    opponent_role_archive_path: Path,
    role_archive_path: Path,
    memories: dict,
    cfg_agent: dict,
    cfg_trainer: dict,
    device: torch.device,
    iteration: int,
    base_archive_path: Path,
    total_iterations: int,
    training_config: SimpleNamespace,
):
    """Initializes, trains, saves, and archives a policy for a given role."""
    print(
        f"\n--- Orchestrating training for {role_to_train_prefix}, Iteration: {iteration + 1} ---"
    )

    models_for_phase = initialize_models_for_mappo(
        env.observation_space,
        env.action_space,
        env.possible_agents,
        device,
        env.get_base_observation_space_structure(),
    )

    agent_instance = MAPPO(
        possible_agents=env.possible_agents,
        models=models_for_phase,
        memories=memories,
        cfg=cfg_agent,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.get_nested_agent_observation_spaces(),
    )

    trained_agent = train_role(
        env,
        agent_instance,
        role_to_train_prefix,
        opponent_role_prefix,
        current_role_checkpoint_path,
        opponent_role_archive_path,
        cfg_trainer,
        tc,
        device,
    )

    new_checkpoint_filename = f"{role_to_train_prefix}_iter_{iteration}_full_agent.pt"
    new_checkpoint_path = base_archive_path / new_checkpoint_filename
    trained_agent.save(str(new_checkpoint_path))
    print(f"Saved {role_to_train_prefix} agent state to: {new_checkpoint_path}")

    if iteration % training_config.archive_save_interval == 0 or iteration == (
        total_iterations - 1
    ):
        add_policy_to_archive(
            str(new_checkpoint_path), role_archive_path, iteration, role_to_train_prefix
        )

    del trained_agent
    del agent_instance
    return str(new_checkpoint_path)


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA errors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)  # Context-manager
    print(f"Using device: {device}")

    # Environment setup
    map_file = Path("maps_templates/lbirinth.json")
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
        cop_archive_path, tc.cop_role_prefix
    )
    current_thief_checkpoint = get_latest_policy_from_archive(
        thief_archive_path, tc.thief_role_prefix
    )

    # Initialize one MAPPO agent instance that will be reconfigured/reloaded.
    # Models will be initialized fresh or loaded inside the loop.
    # SKRL memories for MAPPO
    env.reset()
    memories = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(
            memory_size=CFG_AGENT["rollouts"],
            device=device,
        )

    start_iteration = 13
    total_iterations_to_run = tc.num_self_play_iterations
    end_iteration = start_iteration + total_iterations_to_run

    # ---Main Self-Play Loop---
    for iteration in range(start_iteration, end_iteration):
        print(
            f"\n===== Self-Play Iteration: {iteration + 1}/{end_iteration} (Overall Iteration: {iteration}) ====="
        )

        current_cop_checkpoint = _orchestrate_training_phase(
            env=env,
            role_to_train_prefix=tc.cop_role_prefix,
            opponent_role_prefix=tc.thief_role_prefix,
            current_role_checkpoint_path=current_cop_checkpoint,
            opponent_role_archive_path=thief_archive_path,
            role_archive_path=cop_archive_path,
            memories=memories,
            cfg_agent=CFG_AGENT,
            cfg_trainer=CFG_TRAINER,
            device=device,
            iteration=iteration,
            base_archive_path=base_archive_path,
            total_iterations=end_iteration,
            training_config=tc,
        )

        current_thief_checkpoint = _orchestrate_training_phase(
            env=env,
            role_to_train_prefix=tc.thief_role_prefix,
            opponent_role_prefix=tc.cop_role_prefix,
            current_role_checkpoint_path=current_thief_checkpoint,
            opponent_role_archive_path=cop_archive_path,
            role_archive_path=thief_archive_path,
            memories=memories,
            cfg_agent=CFG_AGENT,
            cfg_trainer=CFG_TRAINER,
            device=device,
            iteration=iteration,
            base_archive_path=base_archive_path,
            total_iterations=end_iteration,
            training_config=tc,
        )

    env.close()
