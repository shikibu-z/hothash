import re
import matplotlib.pyplot as plt


def extract_metrics(filepath):
    reward_pattern = re.compile(r"Average Reward:\s*([0-9.]+)")
    locality_pattern = re.compile(r"Average Locality:\s*([0-9.]+)")
    variance_pattern = re.compile(r"Average Load Variance:\s*([0-9.]+)")

    rewards = []
    locality = []
    load_variance = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if "Average Reward" in lines[i]:
            reward_match = reward_pattern.search(lines[i])
            locality_match = locality_pattern.search(
                lines[i + 1]) if i + 1 < len(lines) else None
            variance_match = variance_pattern.search(
                lines[i + 2]) if i + 2 < len(lines) else None

            if reward_match and locality_match and variance_match:
                rewards.append(float(reward_match.group(1)))
                locality.append(float(locality_match.group(1)))
                load_variance.append(float(variance_match.group(1)))

    return {
        "rewards": rewards,
        "locality": locality,
        "load_variance": load_variance
    }


def plot_training_metrics(training_metrics):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward over time
    axes[0, 0].plot(training_metrics['rewards'])
    axes[0, 0].set_title('Training Reward Progress')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True)

    # Locality score over time
    axes[0, 1].plot(training_metrics['locality'])
    axes[0, 1].set_title('Locality Score Progress')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Locality Score')
    axes[0, 1].grid(True)

    # Load variance over time
    axes[1, 0].plot(training_metrics['load_variance'])
    axes[1, 0].set_title('Load Variance Progress')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Load Variance')
    axes[1, 0].grid(True)

    # Combined metric
    combined_score = [
        0.6 * loc + 0.4 * (1 - var) for loc, var in zip(
            training_metrics['locality'], training_metrics['load_variance'])
    ]
    axes[1, 1].plot(combined_score)
    axes[1, 1].set_title('Combined Performance Score')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Combined Score')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    metrics = extract_metrics("rl.out")
    plot_training_metrics(metrics)
