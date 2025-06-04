import torch
from pathlib import Path
from types import SimpleNamespace
from skrl.multi_agents.torch.mappo import MAPPO
from copy import copy


from utils.policy_archive_utils import (
    add_policy_to_archive,
    get_latest_policy_from_archive,
    sample_policy_from_archive,
)
from utils.model_utils import (
    initialize_models_for_mappo,
    initialize_lstm_models_for_mappo,
    copy_role_models,  # Ensure this is imported
)
from utils.agent_learning_utils import (
    train_role,
    train_simultaneously_and_evaluate,
)  # Added new import
from configs import (
    TrainingConfig,
)


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

    models_for_phase = initialize_lstm_models_for_mappo(
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


def _orchestrate_simultaneous_training_iteration(
    env,
    current_cop_checkpoint: str | None,
    current_thief_checkpoint: str | None,
    cop_archive_path: Path,
    thief_archive_path: Path,
    memories: dict,
    cfg_agent: dict,  # General agent config
    cfg_trainer: dict,
    device: torch.device,
    iteration: int,
    base_archive_path: Path,
    total_iterations: int,
    training_config: SimpleNamespace,
):
    """Initializes, trains both roles simultaneously, saves, and archives the policy."""
    print(
        f"\n--- Orchestrating Simultaneous Training, Iteration: {iteration + 1}/{total_iterations} ---"
    )

    # Initialize a single MAPPO agent instance for this iteration
    models_for_iteration = initialize_lstm_models_for_mappo(
        env.observation_space,
        env.action_space,
        env.possible_agents,
        device,
        env.get_base_observation_space_structure(),
    )

    # Use a general agent configuration
    # Ensure CFG_AGENT is appropriate for combined training
    # If CFG_AGENT_COP and CFG_AGENT_THIEF had significant differences relevant to MAPPO's top-level cfg,
    # you might need to merge them or choose one. For now, we assume CFG_AGENT is suitable.
    agent_instance = MAPPO(
        possible_agents=env.possible_agents,
        models=models_for_iteration,
        memories=memories,
        cfg=cfg_agent,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.get_nested_agent_observation_spaces(),
    )

    # Load cop models if a checkpoint exists
    if current_cop_checkpoint:
        print(f"Loading cop models from: {current_cop_checkpoint}")
        temp_loader_agent_models = initialize_lstm_models_for_mappo(
            env.observation_space,
            env.action_space,
            env.possible_agents,
            device,
            env.get_base_observation_space_structure(),
        )
        # Use a minimal config for the temporary loader agent
        temp_loader_cfg = cfg_agent.copy()
        temp_loader_agent = MAPPO(
            possible_agents=env.possible_agents,
            models=temp_loader_agent_models,
            cfg=temp_loader_cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device,
            shared_observation_spaces=env.get_nested_agent_observation_spaces(),
        )
        temp_loader_agent.load(current_cop_checkpoint)
        copy_role_models(
            temp_loader_agent,
            agent_instance,
            training_config.cop_role_prefix,
            env.possible_agents,
            device,
        )
        del temp_loader_agent

    # Load thief models if a checkpoint exists
    if current_thief_checkpoint:
        print(f"Loading thief models from: {current_thief_checkpoint}")
        # Check if thief checkpoint is different from cop checkpoint to avoid redundant loading if they were the same
        if current_thief_checkpoint != current_cop_checkpoint:
            temp_loader_agent_models = initialize_lstm_models_for_mappo(
                env.observation_space,
                env.action_space,
                env.possible_agents,
                device,
                env.get_base_observation_space_structure(),
            )
            temp_loader_cfg = cfg_agent.copy()
            temp_loader_agent = MAPPO(
                possible_agents=env.possible_agents,
                models=temp_loader_agent_models,
                cfg=temp_loader_cfg,
                observation_spaces=env.observation_spaces,
                action_spaces=env.action_spaces,
                device=device,
                shared_observation_spaces=env.get_nested_agent_observation_spaces(),
            )
            temp_loader_agent.load(current_thief_checkpoint)
            copy_role_models(
                temp_loader_agent,
                agent_instance,
                training_config.thief_role_prefix,
                env.possible_agents,
                device,
            )
            del temp_loader_agent
        elif (
            current_cop_checkpoint
        ):  # Thief checkpoint is same as cop, models already loaded
            print(
                f"Thief models already loaded from shared checkpoint: {current_thief_checkpoint}"
            )

    # Train both roles simultaneously and perform evaluations
    trained_agent = train_simultaneously_and_evaluate(
        env,
        agent_instance,
        cfg_trainer,
        training_config,
        device,
        cop_archive_path,  # For evaluating new thieves against archived cops
        thief_archive_path,  # For evaluating new cops against archived thieves
    )

    # Save the single agent state containing updated policies for both roles
    new_checkpoint_filename = f"joint_iter_{iteration}_full_agent.pt"
    new_checkpoint_path = base_archive_path / new_checkpoint_filename
    trained_agent.save(str(new_checkpoint_path))
    print(f"Saved jointly trained agent state to: {new_checkpoint_path}")

    # Archive this checkpoint for both roles
    if iteration % training_config.archive_save_interval == 0 or iteration == (
        total_iterations - 1
    ):
        add_policy_to_archive(
            str(new_checkpoint_path),
            cop_archive_path,
            iteration,
            training_config.cop_role_prefix,
        )
        add_policy_to_archive(
            str(new_checkpoint_path),
            thief_archive_path,
            iteration,
            training_config.thief_role_prefix,
        )

    del trained_agent
    del agent_instance
    return str(new_checkpoint_path)  # This path is now the latest for both
