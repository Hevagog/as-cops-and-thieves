import torch
from pathlib import Path
from types import SimpleNamespace

from skrl.multi_agents.torch.mappo import MAPPO
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory  # Required for MAPPO initialization

from utils.policy_archive_utils import (
    sample_policy_from_archive,
    update_policy_win_rate,
)
from utils.model_utils import (
    copy_role_models,
    initialize_models_for_mappo,
    initialize_lstm_models_for_mappo,
)
from utils.eval_pfsp_agents import evaluate_agents
from configs import CFG_AGENT, CFG_AGENT_THIEF, CFG_AGENT_COP


def train_role(
    env,
    training_agent: MAPPO,
    role_to_train_prefix: str,
    opponent_role_prefix: str,
    current_role_checkpoint_path: str | None,
    opponent_role_archive_path: Path,
    cfg_trainer: dict,
    training_config: SimpleNamespace,
    device: torch.device,  # device is used for initializing temp_opponent_loader_agent
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
        strategy=training_config.policy_sample_strategy,
    )
    opponent_policy_filename = None
    if opponent_checkpoint_path:
        opponent_policy_filename = Path(opponent_checkpoint_path).name
        print(
            f"Loading opponent ({opponent_role_prefix}) policy from: {opponent_checkpoint_path}"
        )
        # Create a temporary agent to load the opponent's full state
        temp_opponent_loader_agent_models = initialize_lstm_models_for_mappo(
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

        cfg_opponent = (
            CFG_AGENT_COP
            if opponent_role_prefix == training_config.cop_role_prefix
            else CFG_AGENT_THIEF
        ).copy()

        temp_opponent_loader_agent = MAPPO(
            possible_agents=env.possible_agents,
            models=temp_opponent_loader_agent_models,
            memories=temp_memories,
            cfg=cfg_opponent,  # Ensure this CFG is appropriate for a temp loader
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

        # --- Freeze opponent policies for Fictitious Play ---
        print(f"Freezing policies for opponent role: {opponent_role_prefix}")
        for agent_name in training_agent.models:
            if agent_name.startswith(opponent_role_prefix):
                training_agent.models[agent_name]["value"].freeze_parameters()
            elif agent_name.startswith(
                role_to_train_prefix
            ):  # Ensure learning role is trainable
                training_agent.models[agent_name]["policy"].freeze_parameters(False)
                training_agent.models[agent_name]["value"].freeze_parameters(False)
        # --- Freeze policy networks ---
        for agent_name in training_agent.models:
            training_agent.models[agent_name]["policy"].freeze_parameters()

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
            env, training_agent, training_config.n_trial_episodes
        )
        print(f"Avg Cop Return: {avg_cop_return}, Avg Thief Return: {avg_thief_return}")
        if role_to_train_prefix == training_config.cop_role_prefix:
            opponent_won = avg_thief_return > avg_cop_return
        elif role_to_train_prefix == training_config.thief_role_prefix:
            opponent_won = avg_cop_return > avg_thief_return

        update_policy_win_rate(
            opponent_role_archive_path,
            opponent_policy_filename,
            opponent_won,
            training_config.win_rate_buffer_size,
        )

    # --- Random opponent evaluation ---
    if opponent_checkpoint_path:
        evaluate_agent(
            env=env,
            learned_agent=training_agent,
            learned_role_prefix=role_to_train_prefix,
            opponent_role_prefix=opponent_role_prefix,
            opponent_role_archive_path=opponent_checkpoint_path,
            training_config=training_config,
            device=device,
        )

    # We need to return a path to the *newly trained* policy for this role.
    # The training_agent now contains the updated policies for role_to_train
    # AND policies for opponent_role that it trained against.

    return training_agent


def train_simultaneously_and_evaluate(
    env,
    training_agent: MAPPO,
    cfg_trainer: dict,
    training_config: SimpleNamespace,
    device: torch.device,
    cop_archive_path: Path,
    thief_archive_path: Path,
):
    """
    Trains all roles in the training_agent simultaneously, then evaluates them
    against archived policies of the opposing roles.

    training_agent should already have models initialized for all roles!
    """
    print(f"--- Training All Roles Simultaneously ---")

    print(
        "Freezing all policy parameters for simultaneous training. Unfreezing value functions."
    )
    for agent_name in training_agent.models:
        training_agent.models[agent_name]["policy"].freeze_parameters()
        training_agent.models[agent_name]["value"].freeze_parameters(False)

    training_agent.set_mode("train")  # Ensure agent is in training mode

    trainer = SequentialTrainer(env=env, cfg=cfg_trainer, agents=training_agent)
    trainer.train()

    print("\n--- Post-Simultaneous Training Evaluations ---")

    print(
        f"\nEvaluating newly trained {training_config.cop_role_prefix}s against archived {training_config.thief_role_prefix}s."
    )

    evaluate_agent(
        env=env,
        learned_agent=training_agent,
        learned_role_prefix=training_config.cop_role_prefix,
        opponent_role_prefix=training_config.thief_role_prefix,
        opponent_role_archive_path=thief_archive_path,
        training_config=training_config,
        device=device,
    )

    print(
        f"\nEvaluating newly trained {training_config.thief_role_prefix}s against archived {training_config.cop_role_prefix}s."
    )
    evaluate_agent(
        env=env,
        learned_agent=training_agent,
        learned_role_prefix=training_config.thief_role_prefix,
        opponent_role_prefix=training_config.cop_role_prefix,
        opponent_role_archive_path=cop_archive_path,
        training_config=training_config,
        device=device,
    )

    return training_agent


def evaluate_agent(
    env,
    learned_agent: MAPPO,  # Agent with the newly trained policy
    learned_role_prefix: str,  # The role that was just trained
    opponent_role_prefix: str,  # The role of opponents to sample
    opponent_role_archive_path: Path,  # Archive to sample opponents from
    training_config: SimpleNamespace,
    device: torch.device,
    num_additional_opponents_to_evaluate: int = 5,  # Renamed for clarity (default 2, adjust as needed)
) -> None:
    if num_additional_opponents_to_evaluate <= 0:
        return

    print(
        f"\n--- Evaluating {learned_role_prefix} against {num_additional_opponents_to_evaluate} additional opponents from {Path(opponent_role_archive_path).parent} ---"
    )
    # opponent_role_archive_path = Path(opponent_role_archive_path).parent

    # Create a working copy of the agent once.
    # This agent has the learned_role_prefix policy already updated from train_role.
    eval_agent = learned_agent

    # Keep track of opponents evaluated in this function call to try for distinct ones
    evaluated_opponent_filenames_in_this_run = set()

    for i in range(num_additional_opponents_to_evaluate):
        sampled_opponent_checkpoint_path = None
        sampled_opponent_filename = None
        temp_path = None
        # Try to sample a distinct opponent
        for _attempt in range(20):  # Max attempts to find a new one
            temp_path = sample_policy_from_archive(
                opponent_role_archive_path,
                opponent_role_prefix,
                strategy=training_config.policy_sample_strategy,
            )
            if temp_path:
                temp_filename = Path(temp_path).name
                if temp_filename not in evaluated_opponent_filenames_in_this_run:
                    sampled_opponent_checkpoint_path = temp_path
                    sampled_opponent_filename = temp_filename
                    break
            else:  # No policy found by sampling strategy
                break

        if not sampled_opponent_checkpoint_path:
            print(
                f"Could not sample a new distinct additional opponent for evaluation round {i+1}/{num_additional_opponents_to_evaluate}. Using random."
            )
            for _attempt in range(20):  # Max attempts to find a new one
                temp_path = sample_policy_from_archive(
                    opponent_role_archive_path,
                    opponent_role_prefix,
                    strategy="random",
                )
                if temp_path:
                    temp_filename = Path(temp_path).name
                    if temp_filename not in evaluated_opponent_filenames_in_this_run:
                        sampled_opponent_checkpoint_path = temp_path
                        sampled_opponent_filename = temp_filename
                        break
                else:  # No policy found by sampling strategy
                    break

        if not sampled_opponent_checkpoint_path:
            print(
                f"Failed to find a new distinct additional opponent for evaluation round {i+1}/{num_additional_opponents_to_evaluate}. Skipping this evaluation."
            )
            return

        evaluated_opponent_filenames_in_this_run.add(sampled_opponent_filename)
        print(
            f"Loading additional opponent ({opponent_role_prefix}) policy '{sampled_opponent_filename}' from: {sampled_opponent_checkpoint_path}"
        )
        # Load this additional opponent's policy into the eval_agent
        # Create a temporary agent to load the opponent's full state
        temp_opponent_loader_agent_models = initialize_lstm_models_for_mappo(
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
        temp_config = CFG_AGENT.copy()
        temp_config["random_timesteps"] = 0  # No random actions during evaluation
        temp_config["learning_starts"] = 0  # No learning starts during evaluation

        temp_opponent_loader_agent = MAPPO(
            possible_agents=env.possible_agents,
            models=temp_opponent_loader_agent_models,
            memories=temp_memories,
            cfg=temp_config,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device,
            shared_observation_spaces=env.get_nested_agent_observation_spaces(),
        )
        temp_opponent_loader_agent.load(sampled_opponent_checkpoint_path)
        copy_role_models(
            temp_opponent_loader_agent,
            eval_agent,  # Target agent
            opponent_role_prefix,  # Role to copy (the opponent)
            env.possible_agents,
            device,
        )
        del temp_opponent_loader_agent

        for agent_name_eval in eval_agent.models:
            if agent_name_eval.startswith(learned_role_prefix):  # Policy being tested
                eval_agent.models[agent_name_eval]["policy"].freeze_parameters(False)
                eval_agent.models[agent_name_eval]["value"].freeze_parameters(False)
            elif agent_name_eval.startswith(opponent_role_prefix):  # Sampled opponent
                eval_agent.models[agent_name_eval]["policy"].freeze_parameters(True)
                eval_agent.models[agent_name_eval]["value"].freeze_parameters(True)

        add_avg_cop_return, add_avg_thief_return = evaluate_agents(
            env, eval_agent, training_config.n_trial_episodes
        )
        print(
            f"vs Additional Opponent ({sampled_opponent_filename}): Avg Cop Return: {add_avg_cop_return}, Avg Thief Return: {add_avg_thief_return}"
        )

        additional_opponent_policy_won = False
        if (
            learned_role_prefix == training_config.cop_role_prefix
        ):  # Cops were trained, opponent is thief
            if (
                add_avg_thief_return > add_avg_cop_return
            ):  # Thief (additional opponent) won
                additional_opponent_policy_won = True
        elif (
            learned_role_prefix == training_config.thief_role_prefix
        ):  # Thieves were trained, opponent is cop
            if (
                add_avg_cop_return > add_avg_thief_return
            ):  # Cop (additional opponent) won
                additional_opponent_policy_won = True

        update_policy_win_rate(
            opponent_role_archive_path,  # Archive of the opponent role
            sampled_opponent_filename,  # Filename of the opponent policy that was evaluated
            additional_opponent_policy_won,  # Did this opponent win?
            training_config.win_rate_buffer_size,
        )
