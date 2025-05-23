import torch
from skrl.multi_agents.torch.mappo import MAPPO
from environments import BaseEnv
import tqdm


def evaluate_agents(
    env: BaseEnv, agent: MAPPO, n_episodes, cop_prefix="cop", thief_prefix="thief"
):
    """Run n_episodes with all agents frozen, return average returns per role. Fast version."""
    # Freeze all parameters (no grad, no learning)
    for agent_name in agent.models:
        agent.models[agent_name]["policy"].freeze_parameters()
        agent.models[agent_name]["value"].freeze_parameters()
    agent.set_mode("eval")

    cop_win_count = 0
    thief_win_count = 0

    with torch.no_grad():
        total_pbar = tqdm.tqdm(
            total=n_episodes, desc="Evaluating episodes", disable=False
        )
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                actions = {}
                for agent_name in env.possible_agents:
                    obs_tensor = torch.as_tensor(obs[agent_name]).unsqueeze(0)
                    action, *_ = agent.models[agent_name]["policy"].act(
                        {"states": obs_tensor}, role="policy"
                    )
                    action_space = env.action_space(agent_name)
                    if hasattr(action_space, "n"):  # Discrete
                        actions[agent_name] = action.squeeze().to(dtype=torch.long)
                    else:  # Box or others
                        actions[agent_name] = action.squeeze(0).cpu().numpy()
                obs, rewards, terminations, truncations, infos = env.step(actions)
                if any(terminations.values()):
                    # Get winner from info dictionary
                    first_agent = next(iter(infos))
                    winner = infos[first_agent]["winner"]
                    if winner == "cop":
                        cop_win_count += 1
                    elif winner == "thief":
                        thief_win_count += 1
                    done = True

            # Only update progress bar per episode (not per step)
            total_pbar.update(1)
        total_pbar.close()
    env.reset()
    # Unfreeze parameters
    for agent_name in agent.models:
        agent.models[agent_name]["policy"].freeze_parameters(False)
        agent.models[agent_name]["value"].freeze_parameters(False)
    return cop_win_count / n_episodes, thief_win_count / n_episodes
