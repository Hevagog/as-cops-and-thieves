import random
import shutil
import json
from pathlib import Path
from collections import deque

WIN_RATES_FILENAME = "win_rates.json"
DEFAULT_WIN_RATE = 0.5


def load_win_rates(role_archive_path: Path) -> dict:
    """Loads win-rate data from a JSON file in the role's archive directory."""
    win_rates_file = role_archive_path / WIN_RATES_FILENAME
    if win_rates_file.exists():
        with open(win_rates_file, "r") as f:
            try:
                raw_data = json.load(f)
                # Convert lists back to deques
                for policy_name, data in raw_data.items():
                    if "recent_outcomes" in data and isinstance(
                        data["recent_outcomes"], list
                    ):
                        buffer_size = data.get(
                            "buffer_size", 20
                        )  # Default if not stored
                        raw_data[policy_name]["recent_outcomes"] = deque(
                            data["recent_outcomes"], maxlen=buffer_size
                        )
                    if "buffer_size" not in data:  # Ensure buffer_size is present
                        raw_data[policy_name]["buffer_size"] = data.get(
                            "buffer_size", 20
                        )
                return raw_data
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from {win_rates_file}. Returning empty win rates."
                )
                return {}
    return {}


def save_win_rates(role_archive_path: Path, win_rates_data: dict):
    """Saves win-rate data to a JSON file in the role's archive directory."""
    win_rates_file = role_archive_path / WIN_RATES_FILENAME
    # Convert deques to lists for JSON serialization
    serializable_data = {}
    for policy_name, data in win_rates_data.items():
        serializable_data[policy_name] = data.copy()
        if "recent_outcomes" in data and isinstance(data["recent_outcomes"], deque):
            serializable_data[policy_name]["recent_outcomes"] = list(
                data["recent_outcomes"]
            )

    with open(win_rates_file, "w") as f:
        json.dump(serializable_data, f, indent=4)


def update_policy_win_rate(
    role_archive_path: Path,  # Archive path of the policy whose win rate is being updated
    policy_filename: str,  # Filename of the policy, e.g., "cop_iter_0_full_agent.pt"
    won_episode: bool,
    buffer_size: int,
):
    """Updates the win-rate statistics for a given policy."""
    win_rates_data = load_win_rates(role_archive_path)

    if policy_filename not in win_rates_data:
        win_rates_data[policy_filename] = {
            "wins": 0,
            "games": 0,
            "recent_outcomes": deque(maxlen=buffer_size),
            "buffer_size": buffer_size,  # Store buffer_size for consistent deque rehydration
        }

    policy_stats = win_rates_data[policy_filename]
    if policy_stats.get("buffer_size") != buffer_size or not isinstance(
        policy_stats["recent_outcomes"], deque
    ):
        # Re-initialize deque if buffer_size changed or not a deque (e.g. first time after loading old format)
        outcomes_list = list(policy_stats.get("recent_outcomes", []))
        policy_stats["recent_outcomes"] = deque(outcomes_list, maxlen=buffer_size)
        policy_stats["buffer_size"] = buffer_size

    policy_stats["games"] += 1
    if won_episode:
        policy_stats["wins"] += 1
        policy_stats["recent_outcomes"].append(1)
    else:
        policy_stats["recent_outcomes"].append(0)

    save_win_rates(role_archive_path, win_rates_data)
    print(
        f"Updated win rate for {policy_filename} in {role_archive_path}: {'win' if won_episode else 'loss'}. New stats: {policy_stats['wins']}/{policy_stats['games']}"
    )


def add_policy_to_archive(
    checkpoint_path: str,
    role_archive_path: Path,
    iteration_number: int,
    role_prefix: str,
):
    """Copies the checkpoint to the archive, naming it systematically."""
    if not role_archive_path.exists():
        role_archive_path.mkdir(parents=True, exist_ok=True)

    archive_filename = f"{role_prefix}_iter_{iteration_number}.pt"
    destination_path = role_archive_path / archive_filename
    shutil.copy(checkpoint_path, destination_path)
    print(f"Added {checkpoint_path} to archive as {destination_path}")


def get_latest_policy_from_archive(
    role_archive_path: Path, role_prefix: str
) -> str | None:
    """Gets the path to the latest policy for a role based on iteration number."""
    if not role_archive_path.exists():
        return None

    policy_files = list(role_archive_path.glob(f"{role_prefix}_iter_*.pt"))
    if not policy_files:
        return None

    # Extract iteration numbers and find the max
    latest_policy = max(policy_files, key=lambda p: int(p.stem.split("_")[-1]))
    return str(latest_policy)


def sample_policy_from_archive(
    role_archive_path: Path, role_prefix: str, strategy: str = "latest"
) -> str | None:
    """Samples a policy from the archive based on the given strategy."""
    if not role_archive_path.exists():
        return None

    policy_files_paths = list(role_archive_path.glob(f"{role_prefix}_iter_*.pt"))
    if not policy_files_paths:
        return None

    policy_files = [str(p) for p in policy_files_paths]

    if strategy == "latest":
        return get_latest_policy_from_archive(role_archive_path, role_prefix)
    elif strategy == "random":
        return str(random.choice(policy_files))
    elif strategy == "pfsp":
        win_rates_data = load_win_rates(role_archive_path)

        candidate_policies = []
        pfsp_weights = []

        for policy_path_str in policy_files:
            policy_filename = Path(policy_path_str).name
            stats = win_rates_data.get(policy_filename)

            current_win_rate = DEFAULT_WIN_RATE
            if stats and stats["games"] > 0:
                if stats.get("recent_outcomes") and len(stats["recent_outcomes"]) > 0:
                    # Ensure recent_outcomes is a deque or list before sum/len
                    outcomes_iterable = stats["recent_outcomes"]
                    if isinstance(outcomes_iterable, deque) or isinstance(
                        outcomes_iterable, list
                    ):
                        if len(outcomes_iterable) > 0:
                            current_win_rate = sum(outcomes_iterable) / len(
                                outcomes_iterable
                            )
                    else:  # Fallback if it's some other type, though load_win_rates should handle
                        current_win_rate = stats["wins"] / stats["games"]
                else:  # No recent outcomes, use overall
                    current_win_rate = stats["wins"] / stats["games"]

            # PFSP weight: higher for win-rates closer to 0.5
            weight = max(
                1e-3, 1.0 - abs(current_win_rate - 0.5) * 2.0
            )  # Max ensures non-zero probability
            candidate_policies.append(policy_path_str)
            pfsp_weights.append(weight)

        if not candidate_policies:  # Should not happen if policy_files is not empty
            print("Warning: No candidate policies for PFSP, falling back to random.")
            return str(random.choice(policy_files))

        # Normalize weights (optional, random.choices handles unnormalized)
        # total_weight = sum(pfsp_weights)
        # normalized_weights = [w / total_weight for w in pfsp_weights] if total_weight > 0 else None

        # print(f"PFSP Sampling for {role_prefix}: Candidates and Weights:")
        # for cp, cw in zip(candidate_policies, pfsp_weights):
        #     print(f"  {Path(cp).name}: {cw:.3f}")

        selected_policy = random.choices(candidate_policies, weights=pfsp_weights, k=1)[
            0
        ]
        print(f"PFSP selected opponent for {role_prefix}: {Path(selected_policy).name}")
        return selected_policy
    else:
        print(f"Unknown sampling strategy: {strategy}. Defaulting to latest.")
        return get_latest_policy_from_archive(role_archive_path, role_prefix)
