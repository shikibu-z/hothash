import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict, deque
import random
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from exp import Cluster, generate_workload
import matplotlib.pyplot as plt
import hashlib
from typing import Optional
from math import ceil, sqrt
from collections import defaultdict
from pyheaven import TQDM, LoadJson
from pyheaven import HeavenArguments, IntArgumentDescriptor, FloatArgumentDescriptor, LiteralArgumentDescriptor, SwitchArgumentDescriptor, BoolArgumentDescriptor, PrintJson


@dataclass
class Query:
    """Represents a database query"""
    id: str  # Unique identifier for the query
    data_items: List[str]  # Data items this query needs
    op: str
    estimated_cost: float = 1.0  # Processing cost estimate
    priority: int = 1


class ClusterEnvironment:
    """Simulates the cloud database cluster environment"""

    def __init__(self,
                 cluster: Cluster = None,
                 locality_weight: float = 0.8,
                 balance_weight: float = 0.2):
        self.cluster = cluster
        self.num_nodes = cluster.N if cluster else 20  # Default to 20 nodes
        self.data_items = cluster.data_items

        # Initialize data distribution using consistent hashing
        self._initialize_data_distribution()

        # Performance tracking
        self.query_history = deque(maxlen=100000)
        self.load_history = deque(maxlen=100000)
        self.locality_history = deque(maxlen=100000)

        # Optimization goals
        self.locality_weight = locality_weight  # Weight for locality in reward
        self.balance_weight = balance_weight

    def _initialize_data_distribution(self):
        """Initialize data distribution across nodes using consistent hashing"""
        print("Initializing data distribution...")
        for data_item in self.data_items:
            # Use consistent hashing to determine initial placement
            hash_value = int(hashlib.md5(data_item.encode()).hexdigest(), 16)
            node_idx = hash_value % self.num_nodes
            self.cluster.query_node(n=node_idx,
                                    data_id=data_item,
                                    op='mix',
                                    qid=f'init_cache')
        print("Data distribution initialized.")

    def get_state(self, query: Query) -> np.ndarray:
        """Get current environment state for RL agent"""
        state = []

        node_infos = [
            self.cluster.get_node_info(i) for i in range(self.num_nodes)
        ]

        # nodes with query's data
        query_data_distribution = [
            1 if query.data_items[0] in node_info['cached_keys'] else 0
            for node_info in node_infos
        ]
        state.extend(query_data_distribution)

        # Node load information (normalized)
        loads = [self.cluster.get_node_load(i) for i in range(self.num_nodes)]
        max_load = max(loads) if loads and max(loads) > 0 else 1  # 避免除以0
        normalized_loads = [load / max_load for load in loads]
        state.extend(normalized_loads)

        # Use normalized loads to compute variance
        load_variance = np.var(normalized_loads)
        state.append(load_variance)

        # # Data distribution information
        # data_distribution = np.zeros(len(self.data_items))
        # for i, data_item in enumerate(self.data_items):
        #     nodes_with_data = sum(1 for node_info in node_infos
        #                           if data_item in node_info['cached_keys'])
        #     data_distribution[i] = nodes_with_data / self.num_nodes
        # state.extend(data_distribution)

        # Historical performance metrics
        recent_locality = np.mean(list(
            self.locality_history)[-10:]) if self.locality_history else 0
        recent_load_balance = 1 - np.mean(list(
            self.load_history)[-10:]) if self.load_history else 0
        state.extend([recent_locality, recent_load_balance])

        return np.array(state, dtype=np.float32)

    def get_nodes_load(self):
        """Get normalized node loads"""
        loads = [self.cluster.get_node_load(i) for i in range(self.num_nodes)]
        max_load = max(loads) if loads and max(loads) > 0 else 1  # 避免除以0
        normalized_loads = [load / max_load for load in loads]
        return normalized_loads

    def get_valid_actions(self, query: Query) -> List[int]:
        """Get valid node assignments for a query"""
        valid_actions = []

        for i in range(self.num_nodes):
            valid_actions.append(i)

        # for i, node in enumerate(self.num_nodes):
        #     # Check if node has capacity
        #     if node.current_load + query.estimated_cost <= node.max_capacity:
        #         valid_actions.append(i)

        # # Always allow at least one action (best effort)
        # if not valid_actions:
        #     # Find node with minimum load
        #     min_load_idx = min(range(self.num_nodes),
        #                        key=lambda i: self.nodes[i].current_load)
        #     valid_actions.append(min_load_idx)

        return valid_actions

    def calculate_locality_score(self, result: dict) -> float:
        """Calculate data locality score for assigning query to node"""
        if result['cache_miss'] == 1:
            locality_score = 0.0
        else:
            locality_score = 1.0

        return locality_score

    def step(self, query: Query,
             action: int) -> Tuple[float, float, float, bool]:
        """Execute query assignment and return reward"""
        result = self.cluster.query_node(n=action,
                                         data_id=query.data_items[0],
                                         op=query.op,
                                         qid=query.id)

        node_id = self.cluster.node_ids[action]

        # Calculate locality score. 0 for cache miss and 1 for cache hit
        locality_score = self.calculate_locality_score(result)

        # Calculate load balance score
        loads = [self.cluster.get_node_load(i) for i in range(self.num_nodes)]
        max_load = max(loads) if loads and max(loads) > 0 else 1  # 避免除以0
        normalized_loads = [load / max_load for load in loads]
        load_variance = np.var(normalized_loads)
        load_balance_score = 1 / (1 + load_variance)  # Higher is better

        # Combined reward function
        reward = self.locality_weight * locality_score + self.balance_weight * load_balance_score

        # Track metrics
        self.locality_history.append(locality_score)
        self.load_history.append(load_variance)
        self.query_history.append({
            'query_id': query.id,
            'node_id': node_id,
            'locality_score': locality_score,
            'load_balance_score': load_balance_score,
            'reward': reward
        })

        return reward, locality_score, load_balance_score, False  # Not terminal


class DQNNetwork(nn.Module):
    """Deep Q-Network for query scheduling"""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_size: int = 256):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RLQueryScheduler:
    """Reinforcement Learning Query Scheduler"""

    def __init__(self,
                 environment: ClusterEnvironment,
                 learning_rate: float = 0.001):
        self.env = environment
        self.state_size = len(environment.get_state(Query('', [''], 'mix')))
        self.action_size = environment.num_nodes

        # Neural networks
        self.q_network = DQNNetwork(self.state_size, self.action_size)
        self.target_network = DQNNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=learning_rate)

        # Experience replay
        self.memory = deque(maxlen=10000000)
        self.batch_size = 32

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997

        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.target_update_frequency = 7  # Update target network every 7 steps
        self.training_step = 0

    def remember(self, state, action, reward, next_state):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state))

    def act(self, state, valid_actions):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)

        # Mask invalid actions
        masked_q_values = q_values.clone()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q_values[0][i] = float('-inf')

        return masked_q_values.argmax().item()

    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])

        current_q_values = self.q_network(states).gather(
            1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def schedule_query(self, query: Query) -> int:
        """Schedule a query using the trained policy"""
        state = self.env.get_state(query)
        valid_actions = self.env.get_valid_actions(query)
        action = self.act(state, valid_actions)
        return action

    def save_checkpoint(self, filepath="checkpoint.pth"):
        """Save model, memory and training state to file"""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'memory': self.memory,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_step': self.training_step,
            'gamma': self.gamma,
            'target_update_frequency': self.target_update_frequency
        }
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath="checkpoint.pth"):
        """Load model, memory and training state from file"""
        if not os.path.exists(filepath):
            print(f"❌ Checkpoint file {filepath} not found.")
            return

        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        # self.target_network.load_state_dict(checkpoint['target_network'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.memory = checkpoint['memory']
        # self.epsilon = checkpoint['epsilon']
        # self.training_step = checkpoint['training_step']
        print(f"✅ Checkpoint loaded from {filepath}")


class QueryGenerator:
    """Generates realistic query workloads"""

    def __init__(self, workfload: list):
        self.workload = workfload

    def generate_query(self, episode, progress) -> Query:
        """Generate a query with realistic data access patterns"""
        queries = self.workload[episode]
        q, d, o = queries[progress]
        return Query(
            id=q,
            data_items=[d],
            op=o,
        )


def train_scheduler(episodes: int = 1000,
                    num_queries_per_episode: int = 50,
                    cluster: Cluster = None,
                    workload: list = None):
    """Train the RL scheduler"""
    # Initialize environment and components
    env = ClusterEnvironment(cluster=cluster)
    query_generator = QueryGenerator(workload)
    scheduler = RLQueryScheduler(env)

    # Training metrics
    episode_rewards = []
    locality_scores = []
    load_variances = []

    print("Starting training...")

    for episode in range(episodes):
        total_reward = 0
        queries_processed = 0

        # Generate and process queries for this episode
        for i in range(num_queries_per_episode):
            print(f"Episode {episode}, Query {i}/{num_queries_per_episode}")
            query = query_generator.generate_query(episode, i)

            # Get current state
            state = env.get_state(query)
            valid_actions = env.get_valid_actions(query)

            # Choose action
            action = scheduler.act(state, valid_actions)

            # Execute action
            reward, _, _, done = env.step(query, action)
            total_reward += reward
            queries_processed += 1

            # Store experience
            next_state = env.get_state(query)
            scheduler.remember(state, action, reward, next_state)

            # Train the network
            scheduler.replay()

        # Record metrics
        episode_rewards.append(total_reward / queries_processed)

        if env.locality_history:
            locality_scores.append(
                np.mean(list(env.locality_history)[-num_queries_per_episode:]))
        if env.load_history:
            load_variances.append(
                np.mean(list(env.load_history)[-num_queries_per_episode:]))

        # Print progress
        if episode % 1 == 0:
            avg_reward = np.mean(
                episode_rewards[-1:]) if episode_rewards else 0
            avg_locality = np.mean(
                locality_scores[-1:]) if locality_scores else 0
            avg_load_var = np.mean(
                load_variances[-1:]) if load_variances else 0

            print(f"Episode {episode}")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Average Locality: {avg_locality:.4f}")
            print(f"  Average Load Variance: {avg_load_var:.4f}")
            print(f"  Epsilon: {scheduler.epsilon:.4f}")

    return scheduler, env, {
        'rewards': episode_rewards,
        'locality': locality_scores,
        'load_variance': load_variances
    }

    query_generator = QueryGenerator(env.data_items)

    # Reset environment
    for node in env.nodes:
        node.current_load = 0
        node.active_queries = []

    locality_scores = []
    load_variances = []

    print("Evaluating consistent hashing baseline...")

    for i in range(num_queries):
        query = query_generator.generate_query()

        # Use consistent hashing for assignment
        if query.data_items:
            # Hash the first data item to determine node
            hash_value = int(
                hashlib.md5(query.data_items[0].encode()).hexdigest(), 16)
            action = hash_value % env.num_nodes
        else:
            action = 0

        # Check if node has capacity, otherwise use least loaded
        if env.nodes[action].current_load + query.estimated_cost > env.nodes[
                action].max_capacity:
            action = min(range(env.num_nodes),
                         key=lambda i: env.nodes[i].current_load)

        # Execute action
        reward, _ = env.step(query, action)

        # Calculate metrics
        locality_score = env.calculate_locality_score(query, action)
        loads = [n.current_load / n.max_capacity for n in env.nodes]
        load_variance = np.var(loads)

        locality_scores.append(locality_score)
        load_variances.append(load_variance)

        # Simulate completions
        if i % 10 == 0:
            env.simulate_query_completion(completion_rate=0.3)

    print(f"Baseline Results:")
    print(f"  Average Locality Score: {np.mean(locality_scores):.4f}")
    print(f"  Average Load Variance: {np.mean(load_variances):.4f}")
    print(f"  Load Balance Score: {1 - np.mean(load_variances):.4f}")

    return locality_scores, load_variances


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
    args = HeavenArguments.from_parser(descriptors=[
        IntArgumentDescriptor('N', default=20, help='Number of nodes'),
        IntArgumentDescriptor(
            'T', default=180, help='Number of query time steps'),
        IntArgumentDescriptor(
            'q', default=64, help='Average number of queries per time step'),
        IntArgumentDescriptor('D', default=15, help='Number of data items'),
        LiteralArgumentDescriptor('M',
                                  default='ycsb',
                                  choices=['uniform', 'ycsb', 'ssb'],
                                  help='Workload mode'),
        FloatArgumentDescriptor(
            'k', default=1.3, help="Skewness parameter for Zipf's law"),
        IntArgumentDescriptor('s', default=42, help='Random seed'),
        FloatArgumentDescriptor(
            't', default=30.0, help='Duration of each time step in seconds'),
        LiteralArgumentDescriptor(
            'H',
            default='hot',
            choices=['hot', 'balanced', 'bounded', 'cons', 'spore', 'rl'],
            help='Scheduling policy'),
        FloatArgumentDescriptor('a', default=1.00, help='Hot alpha'),
        FloatArgumentDescriptor('e', default=0.300, help='Balanced epsilon'),
        IntArgumentDescriptor('g', default=1, help='SPORE replication factor'),
        IntArgumentDescriptor('r', default=1, help='SPORE threshold'),
        IntArgumentDescriptor('C', default=0, help='Number of node churns'),
        IntArgumentDescriptor(
            'R', default=1, help='Virtual ring randomness control'),
        BoolArgumentDescriptor(
            'O', default=False, help='Dynamic query operation'),
        SwitchArgumentDescriptor('d', default=False, help='debug mode'),
    ])

    print('Initializing cluster and workload...')
    cluster = Cluster(N=args.N,
                      hash_policy=args.H,
                      capacity={
                          'hot': -1,
                          'balanced': 3,
                          'bounded': 3,
                          'cons': -1,
                          'spore': -1,
                          'rl': -1
                      }[args.H],
                      hot_alpha=args.a,
                      num_data=args.D,
                      random=args.R,
                      dyn_op=args.O)

    workload = generate_workload(
        mode=args.M,
        num_time_steps=args.T,
        num_queries_per_time_step=args.q,
        num_data=args.D,
        skewness=args.k,
        seed=args.s,
        dyn_op=args.O,
    )

    # Train the scheduler
    print("Training RL Query Scheduler...")
    print("This may take a few minutes...")
    scheduler, env, training_metrics = train_scheduler(
        episodes=args.T,
        num_queries_per_episode=args.q,
        cluster=cluster,
        workload=workload)

    scheduler.save_checkpoint("rl_scheduler_checkpoint.pth")

    # Plot training progress
    try:
        plot_training_metrics(training_metrics)
    except:
        print("Could not generate plots (matplotlib may not be available)")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    # Analyze node utilization
    # analyze_node_utilization(env)
