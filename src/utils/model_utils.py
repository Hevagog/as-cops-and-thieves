import torch
from skrl.multi_agents.torch.mappo import MAPPO

from models import Policy, Value, LSTMPolicy, LSTMValue


def copy_role_models(
    source_agent: MAPPO,
    target_agent: MAPPO,
    role_prefix: str,
    possible_agents_list: list,
    device: torch.device,
):
    """
    Copies policy and value network state_dicts for a specific role
    from a source MAPPO agent to a target MAPPO agent.
    """
    for agent_name in possible_agents_list:
        if agent_name.startswith(role_prefix):
            if agent_name in source_agent.models and agent_name in target_agent.models:
                # Copy policy
                source_policy_state = source_agent.models[agent_name][
                    "policy"
                ].state_dict()
                target_agent.models[agent_name]["policy"].load_state_dict(
                    source_policy_state
                )
                target_agent.models[agent_name]["policy"].to(device)

                # Copy value
                source_value_state = source_agent.models[agent_name][
                    "value"
                ].state_dict()
                target_agent.models[agent_name]["value"].load_state_dict(
                    source_value_state
                )
                target_agent.models[agent_name]["value"].to(device)
            else:
                print(
                    f"Warning: Agent {agent_name} not found in source or target models for copying."
                )
    print(f"Copied models for role '{role_prefix}' to target agent.")


def initialize_models_for_mappo(
    observation_spaces, action_spaces, possible_agents, device, base_obs_space_struct
) -> dict:
    """Initializes a fresh set of models for the MAPPO agent."""
    models = {}
    for agent_name in possible_agents:
        models[agent_name] = {}
        policy_model = Policy(
            observation_spaces(agent_name),
            action_spaces(agent_name),
            device,
        )
        value_model = Value(
            base_obs_space_struct,
            action_spaces(agent_name),
            device,
            agent_count=len(possible_agents),
        )

        # Compile models if PyTorch version is >= 2.0
        if hasattr(torch, "compile"):
            print(f"Compiling non-LSTM models for {agent_name}...")
            models[agent_name]["policy"] = torch.compile(
                policy_model, mode="reduce-overhead"
            )
            models[agent_name]["value"] = torch.compile(
                value_model, mode="reduce-overhead"
            )
            print(f"Finished compiling non-LSTM models for {agent_name}.")
        else:
            models[agent_name]["policy"] = policy_model
            models[agent_name]["value"] = value_model
            print("torch.compile not available. Using uncompiled non-LSTM models.")

    return models


def initialize_lstm_models_for_mappo(
    observation_spaces,
    action_spaces,
    possible_agents,
    device,
    base_obs_space_struct,
    num_envs=1,
) -> dict:
    """Initializes LSTM models for the MAPPO agent."""
    models = {}
    for agent_name in possible_agents:
        models[agent_name] = {}
        lstm_policy_model = LSTMPolicy(
            observation_spaces(agent_name),
            action_spaces(agent_name),
            device,
            num_envs=num_envs,
        )
        lstm_value_model = LSTMValue(
            base_obs_space_struct,
            action_spaces(agent_name),
            device,
            agent_count=len(possible_agents),
            num_envs=num_envs,
        )

        # Compile models if PyTorch version is >= 2.0
        if hasattr(torch, "compile"):
            print(f"Compiling LSTM models for {agent_name}...")
            models[agent_name]["policy"] = torch.compile(
                lstm_policy_model, mode="reduce-overhead"
            )
            models[agent_name]["value"] = torch.compile(
                lstm_value_model, mode="reduce-overhead"
            )
            print(f"Finished compiling LSTM models for {agent_name}.")
        else:
            models[agent_name]["policy"] = lstm_policy_model
            models[agent_name]["value"] = lstm_value_model
            print("torch.compile not available. Using uncompiled LSTM models.")

    return models
