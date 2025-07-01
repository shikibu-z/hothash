import re
import matplotlib.pyplot as plt


def extract_metrics(filepath):
    reward_pattern = re.compile(r"Average Reward:\s*([0-9.]+)")
    locality_pattern = re.compile(r"Average Locality:\s*([0-9.]+)")
    action_balance_pattern = re.compile(r"Average Action Balance:\s*([0-9.]+)")
    variance_pattern = re.compile(r"Average Load Variance:\s*([0-9.]+)")
    replacement_pattern = re.compile(r"Average Cache Replacement:\s*([0-9.]+)")

    rewards = []
    locality = []
    action_balance = []
    load_variance = []
    cache_replacement = []
    current = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            reward_match = reward_pattern.search(line)
            locality_match = locality_pattern.search(line)
            action_balance_match = action_balance_pattern.search(line)
            variance_match = variance_pattern.search(line)
            replacement_match = replacement_pattern.search(line)

            if reward_match:
                current["reward"] = float(reward_match.group(1))
            if locality_match:
                current["locality"] = float(locality_match.group(1))
            if action_balance_match:
                current["action_balance"] = float(action_balance_match.group(1))
            if variance_match:
                current["load_variance"] = float(variance_match.group(1))
            if replacement_match:
                current["cache_replacement"] = float(replacement_match.group(1))

                rewards.append(current.get("reward", 0))
                locality.append(current.get("locality", 0))
                action_balance.append(current.get("action_balance", 0))
                load_variance.append(current.get("load_variance", 0))
                cache_replacement.append(current.get("cache_replacement", 0))
                current = {}

    return {
        "rewards": rewards,
        "locality": locality,
        "action_balance": action_balance,
        "load_variance": load_variance,
        "cache_replacement": cache_replacement,
    }


def plot_training_metrics(training_metrics):
    """Plot training progress"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Reward over time
    axes[0].plot(training_metrics["rewards"])
    axes[0].set_title("Training Reward Progress")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Reward")
    axes[0].grid(True)

    # Locality over time
    axes[1].plot(training_metrics["locality"])
    axes[1].set_title("Locality Score Progress")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average Locality")
    axes[1].grid(True)

    # Action balance
    axes[2].plot(training_metrics["action_balance"])
    axes[2].set_title("Action Balance Progress")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Average Action Balance")
    axes[2].grid(True)

    # Load variance
    axes[3].plot(training_metrics["load_variance"])
    axes[3].set_title("Load Variance Progress")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Average Load Variance")
    axes[3].grid(True)

    # Cache replacement
    axes[4].plot(training_metrics["cache_replacement"])
    axes[4].set_title("Cache Replacement Progress")
    axes[4].set_xlabel("Episode")
    axes[4].set_ylabel("Average Cache Replacement")
    axes[4].grid(True)

    # Combined score
    combined_score = [
        0.6 * loc + 0.4 * (1 - var)
        for loc, var in zip(
            training_metrics["locality"], training_metrics["load_variance"]
        )
    ]
    axes[5].plot(combined_score)
    axes[5].set_title("Combined Performance Score")
    axes[5].set_xlabel("Episode")
    axes[5].set_ylabel("Combined Score")
    axes[5].grid(True)

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    metrics = extract_metrics("rl.out")
    plot_training_metrics(metrics)
