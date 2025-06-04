import torch
from pathlib import Path
import os
from copy import copy
from types import SimpleNamespace

from environments import SimpleEnv
from maps import Map
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

from utils.policy_archive_utils import (
    sample_policy_from_archive,
)
from training import _orchestrate_simultaneous_training_iteration
from configs import (
    CFG_AGENT,
    CFG_TRAINER,
    TrainingConfig,
)

tc = copy(TrainingConfig)


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA errors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)  # Context-manager
    print(f"Using device: {device}")

    # Environment setup
    map_file = Path("maps_templates/squarinth.json")
    game_map = Map(map_file)
    env = SimpleEnv(map=game_map, render_mode="rgb_array", max_step_count=2000)
    env = wrap_env(env, wrapper="pettingzoo")

    # Policy Archive Setup
    base_archive_path = Path(
        "lstm_policy_archive_self_play_new"
    )  # Changed archive name
    cop_archive_path = base_archive_path / "cops"
    thief_archive_path = base_archive_path / "thieves"
    cop_archive_path.mkdir(parents=True, exist_ok=True)
    thief_archive_path.mkdir(parents=True, exist_ok=True)

    # Get the latest *joint* checkpoint. If archives are from previous separate trainings,
    # this might need careful handling for the first run.
    # For simplicity, we assume that if a joint checkpoint exists, it's used for both.
    # Otherwise, try to get latest individual ones.
    latest_joint_checkpoint_cop = sample_policy_from_archive(
        cop_archive_path, tc.cop_role_prefix
    )
    latest_joint_checkpoint_thief = sample_policy_from_archive(
        thief_archive_path, tc.thief_role_prefix
    )

    current_cop_checkpoint = latest_joint_checkpoint_cop
    current_thief_checkpoint = latest_joint_checkpoint_thief

    if not current_cop_checkpoint:
        current_cop_checkpoint = sample_policy_from_archive(
            cop_archive_path, tc.cop_role_prefix
        )
        print(
            f"No joint cop checkpoint found, using latest cop-specific: {current_cop_checkpoint}"
        )
    if not current_thief_checkpoint:
        current_thief_checkpoint = sample_policy_from_archive(
            thief_archive_path, tc.thief_role_prefix
        )
        print(
            f"No joint thief checkpoint found, using latest thief-specific: {current_thief_checkpoint}"
        )

    env.reset()
    memories = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(
            memory_size=CFG_AGENT["rollouts"],
            device=device,
        )

    start_iteration = 15
    total_iterations_to_run = tc.num_self_play_iterations
    end_iteration = start_iteration + total_iterations_to_run
    general_cfg_trainer = copy(CFG_TRAINER)

    # ---Main Self-Play Loop---
    for iteration in range(start_iteration, end_iteration):
        print(
            f"\n===== Self-Play Iteration: {iteration + 1}/{end_iteration} (Overall Iteration: {iteration}) ====="
        )

        # A single checkpoint path is returned, representing the new state for both roles
        latest_checkpoint = _orchestrate_simultaneous_training_iteration(
            env=env,
            current_cop_checkpoint=current_cop_checkpoint,
            current_thief_checkpoint=current_thief_checkpoint,
            cop_archive_path=cop_archive_path,
            thief_archive_path=thief_archive_path,
            memories=memories,
            cfg_agent=CFG_AGENT,
            cfg_trainer=general_cfg_trainer,
            device=device,
            iteration=iteration,
            base_archive_path=base_archive_path,
            total_iterations=end_iteration,
            training_config=tc,
        )
        current_cop_checkpoint = sample_policy_from_archive(
            cop_archive_path, tc.cop_role_prefix, strategy="latest"
        )
        current_thief_checkpoint = sample_policy_from_archive(
            thief_archive_path, tc.thief_role_prefix, strategy="latest"
        )

    env.close()
