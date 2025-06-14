rl.out 日志内容格式如下:
Episode 29, Query 1/2
Episode 29
  Average Reward: 0.0865
  Average Locality: 0.0000
  Average Action Balance: 0.4167
  Average Load Variance: 0.2671
  Average Cache Replacement: 0.0000
  Epsilon: 0.8886
Episode 30, Query 0/2
Episode 30, Query 1/2

原始代码如下，帮我改进以能够提取所有新的 Average Reward: 0.0865
  Average Locality: 0.0000
  Average Action Balance: 0.4167
  Average Load Variance: 0.2671
  Average Cache Replacement: 0.0000


代码：
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



我的训练逻辑如下，但是训练出来的模型总是倾向于把大部分 query schedule到少数node上，这是为什么
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


实际的 cloud 环境 train 十分缓慢, 但对cloud 的需求就是query_node和get_node_load，所以我们可以用一个 simulation 环境来测试。
result = self.cluster.query_node(n=action,data_id=query.data_items[0], op=query.op, qid=query.id)
load = self.cluster.get_node_load(i)

query_node的返回result 关键为：
'cache_miss': cache_miss,
'cache_replacement': cache_replacement

get_node_load(i)得到 cluster中 node i的历史累计 query 数量

帮我写一个simulation 的 cluster 类，nodes 数量为20，每个 nodes 上的 cache size 为 8，cache 采用 FIFO 替换策略




再添加一个类函数get_node_info(i)，目的是为了获得 node i 上cached key信息,返回结果为
{
"cached_keys": {
            key: True
            for key in CACHE.keys()
}
}


我不在希望一次性generate_workload,而是希望有一个类，generate_workload，通过类函数generate_query每次生成一条能够生成符合mode（例如choices=['uniform', 'ycsb', 'ssb','ycsb'])的 query 出来，query 的格式要符合generate_workload的输出，帮我改造QueryGenerator

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


def generate_workload(
    mode="ycsb",
    num_time_steps=10,
    num_queries_per_time_step=100,
    num_data=15,
    skewness=1.01,
    seed=42,
    dyn_op=False,
):
    np.random.seed(seed)
    data = list()
    query_id = 0
    lodate = get_lodate(num_data, B=7)

    for t in range(num_time_steps):
        num_queries = num_queries_per_time_step
        data_time_step = list()
        if mode == 'uniform':
            data_ids = np.random.choice(num_data, num_queries, replace=True)
        elif mode == "ycsb":
            data_ids = [
                Hash(x) % num_data
                for x in np.random.zipf(skewness, num_queries)
            ]
        elif mode == "ssb":
            bucket_ids = [
                Hash(x) % 7 for x in np.random.zipf(skewness, num_queries)
            ]
            all_data_ids = [
                np.random.choice(lodate[bucket_id]) for bucket_id in bucket_ids
                for _ in range(max(ceil(0.2 * len(lodate[bucket_id])), 1))
            ]
            data_ids = np.random.choice(all_data_ids,
                                        num_queries,
                                        replace=False)
        for i in range(num_queries):
            data_time_step.append(
                (f"Q{query_id:06d}", f"D{data_ids[i]:03d}", 'mix'))
            query_id += 1

        data.append(data_time_step)

    # dynamic query operation
    if dyn_op:
        ops = ['sum', 'mean', 'min', 'max']
        keys = [f"D{d:03d}" for d in range(num_data)]
        key2op = dict()
        for key in keys:
            key2op[key] = np.random.choice(ops)
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = (data[i][j][0], data[i][j][1],
                              key2op[data[i][j][1]])

    return data





    /data/vldb2025/hothash/gcp/rl.py:381: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:254.)
  states = torch.FloatTensor([e[0] for e in batch])



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

帮我 fix