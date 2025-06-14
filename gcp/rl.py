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
from exp import generate_workload
import matplotlib.pyplot as plt
import hashlib
from typing import Optional
from math import ceil, sqrt
from collections import defaultdict
from pyheaven import TQDM, LoadJson
from collections import Counter
from pyheaven import HeavenArguments, IntArgumentDescriptor, FloatArgumentDescriptor, LiteralArgumentDescriptor, SwitchArgumentDescriptor, BoolArgumentDescriptor, PrintJson
from collections import deque, defaultdict
import random


def Hash(key, salt=None):
    return int(
        hashlib.md5((f"{key}" if salt is None else
                     f"{key}|{salt}").encode('utf-8')).hexdigest(), 16)


def get_lodate(num_data, B=7):
    bucket_size = num_data // B
    bucket = [[] for _ in range(B)]
    for data_id in range(num_data):
        bucket_index = min(int(data_id / bucket_size), 6)
        bucket[bucket_index].append(data_id)
    return bucket


@dataclass
class Query:
    """Represents a database query"""
    id: str  # Unique identifier for the query
    data_items: List[str]  # Data items this query needs
    op: str
    estimated_cost: float = 1.0  # Processing cost estimate
    priority: int = 1


class Cluster:

    def __init__(self, num_nodes=20, cache_size=8, num_data=15):
        self.num_nodes = num_nodes
        self.cache_size = cache_size

        # 每个节点的 cache，用 deque 实现 FIFO
        self.caches = [deque(maxlen=cache_size) for _ in range(num_nodes)]

        # 每个节点累计 query 次数
        self.node_query_count = [0 for _ in range(num_nodes)]

        self.data_items = [f"D{i:03d}" for i in range(num_data)]
        self.node_ids = [
            f"hothash-node-{i:02d}" for i in range(self.num_nodes)
        ]

    def query_node(self, n, data_id, op=None, qid=None):
        """
        Simulate querying a node.
        
        Args:
            n: node index
            data_id: queried data ID
            op, qid: not used, kept for compatibility
        
        Returns:
            dict with cache_miss and cache_replacement flags
        """
        cache = self.caches[n]

        if qid != 'init_cache':
            self.node_query_count[n] += 1

        cache_miss = 0
        cache_replacement = 0

        if data_id in cache:
            # Cache hit: move to end to simulate FIFO freshness
            cache.remove(data_id)
            cache.append(data_id)
        else:
            cache_miss = 1
            if len(cache) >= self.cache_size:
                cache_replacement = 1
            cache.append(data_id)

        return {
            'cache_miss': cache_miss,
            'cache_replacement': cache_replacement
        }

    def get_node_load(self, i):
        """
        Get the historical cumulative query count of node i.
        """
        if 0 <= i < self.num_nodes:
            return self.node_query_count[i]
        else:
            raise IndexError(
                f"Node index {i} out of range (0-{self.num_nodes - 1})")

    def get_node_info(self, i):
        if 0 <= i < self.num_nodes:
            cache = self.caches[i]
            return {"cached_keys": {key: True for key in cache}}
        else:
            raise IndexError(
                f"Node index {i} out of range (0-{self.num_nodes - 1})")

    def reset(self):
        """
        Reset the cluster to initial state.
        """
        self.caches = [
            deque(maxlen=self.cache_size) for _ in range(self.num_nodes)
        ]
        self.node_query_count = [0 for _ in range(self.num_nodes)]


class ClusterEnvironment:
    """Simulates the cloud database cluster environment"""

    def __init__(
        self,
        cluster: Cluster = None,
        cache_size: int = 8,
        locality_weight: float = 0.5,
        balance_weight: float = 0.5,
        imbalance_weight: float = 0.5,
        replacement_wight: float = 1.0,
    ):
        self.cluster = cluster
        self.num_nodes = cluster.num_nodes if cluster else 20  # Default to 20 nodes
        self.data_items = cluster.data_items
        self.cache_size = cache_size

        # Initialize data distribution using consistent hashing
        self._initialize_data_distribution()

        # Performance tracking
        self.query_history = deque(maxlen=10000000)
        self.locality_history = deque(maxlen=10000000)
        self.action_balance_history = deque(maxlen=10000000)
        self.global_load_imbalance_history = deque(maxlen=10000000)
        self.cache_replacement_history = deque(maxlen=10000000)
        self.action_history = deque(maxlen=10000000)

        # Optimization goals
        self.locality_weight = locality_weight
        self.balance_weight = balance_weight
        self.imbalance_weight = imbalance_weight
        self.replacement_weight = replacement_wight

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

        # nodes without query's data and is full
        node_cache_full = [
            1 if query_data_distribution[i] == 0
            and len(node_info['cached_keys']) >= self.cache_size else 0
            for i, node_info in enumerate(node_infos)
        ]
        state.extend(node_cache_full)

        # Node load information (normalized)
        normalized_loads = self.get_nodes_load()
        state.extend(normalized_loads)

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

        return valid_actions

    def calculate_locality_score(self, result: dict) -> float:
        """Calculate data locality score for assigning query to node"""
        if result['cache_miss'] == 1:
            locality_score = 0.0
        else:
            locality_score = 1.0

        return locality_score

    def get_cache_replacement(self, result: dict) -> float:
        """Get cache replacement from query result"""
        if result['cache_replacement'] == 1:
            cache_replacement = 1.0
        else:
            cache_replacement = 0.0

        return cache_replacement

    def step(self, query: Query,
             action: int) -> Tuple[float, float, float, bool]:
        """Execute query assignment and return reward"""
        # Get node loads before executing the action
        normalized_loads = self.get_nodes_load()

        result = self.cluster.query_node(n=action,
                                         data_id=query.data_items[0],
                                         op=query.op,
                                         qid=query.id)

        node_id = self.cluster.node_ids[action]

        # Calculate locality score. 0 for cache miss and 1 for cache hit
        locality_score = self.calculate_locality_score(result)

        # Calculate action balance score [0-1]
        chosen_node_load = normalized_loads[action]
        action_balance_score = 1 - chosen_node_load

        # 3. global load imbalance penalty
        global_load_std = np.std(normalized_loads)
        imbalance_penalty = -self.imbalance_weight * global_load_std

        # 4. cache replacement penalty
        replacement_penalty = -self.replacement_weight * self.get_cache_replacement(
            result)

        # 5. final reward calculation.
        reward = (self.locality_weight * locality_score +
                  self.balance_weight * action_balance_score +
                  imbalance_penalty + replacement_penalty)

        # Track metrics
        self.locality_history.append(locality_score)
        self.action_balance_history.append(action_balance_score)
        self.global_load_imbalance_history.append(global_load_std)
        self.cache_replacement_history.append(
            self.get_cache_replacement(result))
        self.query_history.append({
            'query_id':
            query.id,
            'node_id':
            node_id,
            'data_item':
            query.data_items[0],
            'locality_score':
            locality_score,
            'action_balance_score':
            global_load_std,
            'cache_replacement':
            self.get_cache_replacement(result),
            'reward':
            reward
        })

        return reward, False  # Not terminal


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
        self.memory = deque(maxlen=100000)
        self.batch_size = 256

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995

        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.target_update_frequency = 5000  # Update target network every 5000 steps
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

        # 从经验池中随机采样一个批次
        batch = random.sample(self.memory, self.batch_size)

        # 先用 np.array() 将列表转换为单个 numpy 数组，再转换为张量
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor(np.array([e[1] for e in batch]))
        rewards = torch.FloatTensor(np.array([e[2] for e in batch]))
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))

        # 获取当前状态的 Q 值
        current_q_values = self.q_network(states).gather(
            1, actions.unsqueeze(1))

        # 获取下一个状态的最大 Q 值（注意使用 target_network）
        next_q_values = self.target_network(next_states).max(1)[0].detach()

        # 计算目标 Q 值 (Bellman 方程)
        target_q_values = rewards + (self.gamma * next_q_values)

        # 计算损失函数 (MSE Loss)
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率 (epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期更新目标网络
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
        print(f"✅ Checkpoint loaded from {filepath}")


class QueryGenerator:
    """Generates realistic query workloads dynamically"""

    def __init__(self,
                 mode="ycsb",
                 num_data=15,
                 skewness=1.3,
                 seed=42,
                 dyn_op=False):
        """
        Initialize the QueryGenerator
        
        Args:
            mode: Query generation mode, choices=['uniform', 'ycsb', 'ssb']
            num_data: Number of data items
            skewness: Skewness parameter for zipf distribution
            seed: Random seed
            dyn_op: Whether to use dynamic operations
        """
        self.mode = mode
        self.num_data = num_data
        self.skewness = skewness
        self.seed = seed
        self.dyn_op = dyn_op
        self.query_counter = 0

        # Set random seed
        np.random.seed(seed)

        # Initialize data structures based on mode
        if mode == "ssb":
            self.lodate = get_lodate(num_data, B=7)

        # Initialize operations for dynamic operation mode
        if dyn_op:
            self.ops = ['sum', 'mean', 'min', 'max']
            self.keys = [f"D{d:03d}" for d in range(num_data)]
            self.key2op = {
                key: np.random.choice(self.ops)
                for key in self.keys
            }

    def generate_query(self) -> Query:
        """Generate a single query based on the configured mode"""
        # Generate data_id based on mode
        if self.mode == 'uniform':
            data_id = np.random.choice(self.num_data)
        elif self.mode == "ycsb":
            data_id = Hash(np.random.zipf(self.skewness)) % self.num_data
        elif self.mode == "ssb":
            bucket_id = Hash(np.random.zipf(self.skewness)) % 7
            # Select from the bucket with some probability
            bucket_data = self.lodate[bucket_id]
            if bucket_data:
                data_id = np.random.choice(bucket_data)
            else:
                data_id = np.random.choice(self.num_data)  # fallback
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Generate query components
        query_id = f"Q{self.query_counter:06d}"
        data_item = f"D{data_id:03d}"

        # Determine operation
        if self.dyn_op:
            op = self.key2op[data_item]
        else:
            op = 'mix'

        # Increment counter
        self.query_counter += 1

        return Query(
            id=query_id,
            data_items=[data_item],
            op=op,
        )

    def generate_batch_queries(self, num_queries: int) -> List[Query]:
        """Generate a batch of queries"""
        return [self.generate_query() for _ in range(num_queries)]

    def reset_counter(self):
        """Reset the query counter"""
        self.query_counter = 0

    def set_seed(self, seed: int):
        """Set a new random seed"""
        self.seed = seed
        np.random.seed(seed)


def train_scheduler(episodes: int = 1000,
                    num_queries_per_episode: int = 50,
                    cluster: Cluster = None):
    """Train the RL scheduler"""
    # Initialize environment and components
    env = ClusterEnvironment(cluster=cluster)
    query_generator = QueryGenerator()
    scheduler = RLQueryScheduler(env)

    # Training metrics
    episode_rewards = []
    locality_scores = []
    action_balance_scores = []
    global_load_imbalance = []
    cache_replacement = []

    print("Starting training...")

    for episode in range(episodes):
        total_reward = 0
        queries_processed = 0

        # Generate and process queries for this episode
        for i in range(num_queries_per_episode):
            query = query_generator.generate_query()

            # Get current state
            state = env.get_state(query)
            valid_actions = env.get_valid_actions(query)

            # Choose action
            action = scheduler.act(state, valid_actions)
            env.action_history.append(action)

            # Execute action
            reward, done = env.step(query, action)
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
        if env.action_balance_history:
            action_balance_scores.append(
                np.mean(
                    list(env.action_balance_history)
                    [-num_queries_per_episode:]))
        if env.global_load_imbalance_history:
            global_load_imbalance.append(
                np.mean(
                    list(env.global_load_imbalance_history)
                    [-num_queries_per_episode:]))
        if env.cache_replacement_history:
            cache_replacement = [
                np.var(
                    list(env.cache_replacement_history)
                    [-num_queries_per_episode:])
            ]

        # Print progress
        if episode % 1 == 0:
            avg_reward = np.mean(
                episode_rewards[-1:]) if episode_rewards else 0
            avg_locality = np.mean(
                locality_scores[-1:]) if locality_scores else 0
            avg_action_balance = np.mean(
                action_balance_scores[-1:]) if action_balance_scores else 0
            avg_load_var = np.mean(
                global_load_imbalance[-1:]) if global_load_imbalance else 0
            avg_cache_replacement = np.mean(
                cache_replacement[-1:]) if cache_replacement else 0

            print(f"Episode {episode}")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Average Locality: {avg_locality:.4f}")
            print(f"  Average Action Balance: {avg_action_balance:.4f}")
            print(f"  Average Load Variance: {avg_load_var:.4f}")
            print(f"  Average Cache Replacement: {avg_cache_replacement:.4f}")
            print(f"  Epsilon: {scheduler.epsilon:.4f}")

            # Check for convergence criteria
            if episode > 600 and avg_locality > 0.999 and avg_cache_replacement < 0.001:
                return scheduler, env, {
                    'rewards': episode_rewards,
                    'locality': locality_scores,
                    'action_balance': action_balance_scores,
                    'load_variance': global_load_imbalance,
                    'cache_replacement': cache_replacement,
                    'action_history': list(env.action_history)
                }

    return scheduler, env, {
        'rewards': episode_rewards,
        'locality': locality_scores,
        'action_balance': action_balance_scores,
        'load_variance': global_load_imbalance,
        'cache_replacement': cache_replacement,
        'action_history': list(env.action_history)
    }


def plot_training_metrics(training_metrics):
    """Plot training progress with extended metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

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

    # Action balance over time
    axes[1, 0].plot(training_metrics['action_balance'])
    axes[1, 0].set_title('Action Balance Progress')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Action Balance Score')
    axes[1, 0].grid(True)

    # Load variance over time
    axes[1, 1].plot(training_metrics['load_variance'])
    axes[1, 1].set_title('Load Variance Progress')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Load Variance')
    axes[1, 1].grid(True)

    # Cache replacement over time
    axes[2, 0].plot(training_metrics['cache_replacement'])
    axes[2, 0].set_title('Cache Replacement Events')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Replacement Count')
    axes[2, 0].grid(True)

    # Combined performance score
    combined_score = [
        0.4 * loc + 0.3 * (1 - var) + 0.3 * bal for loc, var, bal in zip(
            training_metrics['locality'], training_metrics['load_variance'],
            training_metrics['action_balance'])
    ]
    axes[2, 1].plot(combined_score)
    axes[2, 1].set_title('Combined Performance Score')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Combined Score')
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_action_history(action_history):
    """Plot action distribution as a bar chart"""
    # 统计每个动作的出现次数
    action_counts = Counter(action_history)

    # 提取动作和对应频次
    actions = list(action_counts.keys())
    counts = [action_counts[a] for a in actions]

    # 画柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(actions, counts, color='skyblue', edgecolor='black')
    plt.title('Action Distribution Over Episodes')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('action_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    args = HeavenArguments.from_parser(descriptors=[
        IntArgumentDescriptor('N', default=20, help='Number of nodes'),
        IntArgumentDescriptor(
            'T', default=1000, help='Number of query time steps'),
        IntArgumentDescriptor(
            'q', default=128, help='Average number of queries per time step'),
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
            default='rl',
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
    cluster = Cluster()
    cluster.reset()

    # Train the scheduler
    print("Training RL Query Scheduler...")
    print("This may take a few minutes...")
    scheduler, env, training_metrics = train_scheduler(
        episodes=args.T, num_queries_per_episode=args.q, cluster=cluster)

    scheduler.save_checkpoint("rl_scheduler_checkpoint_v2.pth")

    # Plot training progress
    try:
        plot_training_metrics(training_metrics)
        plot_action_history(training_metrics['action_history'])
    except:
        print("Could not generate plots (matplotlib may not be available)")

    print('Cache hit rate: {:.2f}%'.format(
        env.locality_history.count(1) / len(env.locality_history) * 100))

    print('Average action balance score: {:.4f}'.format(
        sum(env.action_balance_history) / len(env.action_balance_history)))

    print('Average global load imbalance: {:.4f}'.format(
        sum(env.global_load_imbalance_history) /
        len(env.global_load_imbalance_history)))

    print('Cache replacement rate: {:.4f}%'.format(
        env.cache_replacement_history.count(1) /
        len(env.cache_replacement_history) * 100))

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
