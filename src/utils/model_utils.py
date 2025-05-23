import torch
from skrl.multi_agents.torch.mappo import MAPPO

from models import Policy, Value


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
        models[agent_name]["policy"] = Policy(
            observation_spaces(agent_name),
            action_spaces(agent_name),
            device,
        )
        models[agent_name]["value"] = Value(
            base_obs_space_struct,  # Use the shared structure
            action_spaces(agent_name),
            device,
            agent_count=len(possible_agents),
        )
    return models
