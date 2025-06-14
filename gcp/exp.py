import sys
import time
import json
import bisect
import hashlib
import requests
import threading
import concurrent.futures
from math import ceil, sqrt
from collections import defaultdict
from pyheaven import TQDM, LoadJson
from collections import defaultdict, deque
from rl_scheduler import Query, RLQueryScheduler
from typing import Dict, List, Tuple, Optional
import numpy as np
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def Hash(key, salt=None):
    return int(
        hashlib.md5((f"{key}" if salt is None else
                     f"{key}|{salt}").encode('utf-8')).hexdigest(), 16)


class Cluster:

    def __init__(self,
                 N=20,
                 hash_policy='hot',
                 capacity=-1,
                 hot_alpha=1.0,
                 num_data=15,
                 random=1,
                 dyn_op=False,
                 debug=False):
        self.config = json.load(open('config.json'))
        self.hash_policy = hash_policy
        self.capacity = capacity
        self.alpha = hot_alpha
        self.debug = debug
        self.dyn_op = dyn_op
        self.rl_agent: RLQueryScheduler = None  # RL agent for dynamic scheduling
        self.init(N, num_data, random)

    def get_node_api(self, n):
        node_internal_ip = self.config['node_internal_ips'][
            f'hothash-node-{n+1:02d}']
        return f"http://{node_internal_ip}:8080"

    def query_node(self, n, data_id, op='mix', qid=''):
        try:
            if self.dyn_op:
                self.node_loads[n] += (self.dyn_stats[data_id] /
                                       self.dyn_bound[data_id])
            else:
                self.node_loads[n] += 1

            if qid == 'init_cache':
                self.node_loads[n] -= 1

            if self.debug:
                return {
                    "result": 0,
                    'tot_server_time': 0,
                    'fetch_time': 0,
                    'compute_time': 0,
                    'data_id': 0,
                    'node_id': 0,
                    'query_id': 0,
                    'query_op': None
                }
            return requests.get(self.get_node_api(n) + '/query',
                                params={
                                    'op': op,
                                    'data_id': data_id,
                                    'query_id': qid
                                }).json()
        except Exception as e:
            print(n, data_id, e)
            return None

    def clear_node(self, n):
        try:
            self.node_loads[n] = 0
            if self.debug:
                return
            requests.get(self.get_node_api(n) + '/clear_cache')
        except Exception as e:
            print(n, e)

    def get_node_info(self, n):
        try:
            if self.debug:
                return {
                    "version": 0,
                    "num_queries": 0,
                    "num_replicas": 0,
                    "cache_size": 0,
                    "cached_keys": list()
                }
            return requests.get(self.get_node_api(n) + '/info').json()
        except Exception as e:
            print(n, e)
            return None

    def node_churn(self, n, status="DISABLE"):
        if status == "DISABLE":
            self.disabled.add(n)
        elif status == "ENABLE":
            self.disabled.discard(n)

    def init(self, N, num_data, random):
        self.N = N
        self.num_nodes = N
        self.cache_size = 8  # default cache size
        self.cost = {'num_queries': 0, 'cache_hit': 0, 'replica': 0}
        self.hash_rings = {}
        self.node_loads = defaultdict(int)
        self.stats = defaultdict(int)
        for n in range(N):
            self.clear_node(n)

        # node churn
        self.disabled = set()

        # virtual ring randomness
        self.vring_cost = {'size': 0, 'time': 0}
        init_start = time.time()
        self.vring_map = dict()
        self.data_items = [f"D{i:03d}" for i in range(num_data)]
        self.node_ids = [f"hothash-node-{i:02d}" for i in range(self.N)]
        data_ids = [f"D{i:03d}" for i in range(num_data)]
        while len(data_ids) > 0:
            temp = list()
            for _ in range(random):
                if len(data_ids) != 0:
                    idx = np.random.randint(0, len(data_ids))
                    temp.append(data_ids[idx])
                    data_ids.pop(idx)
            for key in temp:
                self.vring_map[key] = temp[0]
        self.vring_cost['size'] += self.total_size(self.vring_map)
        self.vring_cost['time'] += (time.time() - init_start)

        # dynamic query operation
        self.dyn_op_cost = {'size': 0, 'time': 0}
        init_start = time.time()
        if self.dyn_op:
            log_path = f"./profile_DYNOP_D{num_data}_hot.json"
            query_log = LoadJson(log_path)[1]
            self.dyn_stats = defaultdict(int)
            self.dyn_bound = defaultdict(int)
            for _, value in query_log.items():
                self.dyn_stats[value["data_id"]] += value["compute_time"]
                self.dyn_bound[value["data_id"]] += 1
            self.dyn_cost = {
                "num_queries":
                sum(value for _, value in self.dyn_stats.items()),
                "cache_hit": 0,
                "replica": 0,
            }
            self.dyn_op_cost['size'] += self.total_size(self.dyn_stats)
            self.dyn_op_cost['size'] += self.total_size(self.dyn_cost)
        self.dyn_op_cost['time'] += (time.time() - init_start)

    def get_node_load(self, n):
        return self.node_loads[n]
        # return self.get_node_info(n)['num_queries']

    def _get_hash_ring(self, salt=None):
        if salt in self.hash_rings: return self.hash_rings[salt]
        self.hash_rings[salt] = sorted([(Hash(i, salt), i)
                                        for i in range(self.N)])
        return self.hash_rings[salt]

    # node churn
    def get_hash_node_churn(self, key, salt=None):
        hash_ring = self._get_hash_ring(salt)
        hash_key = Hash(key, salt)
        idx = bisect.bisect_right(hash_ring, x=(hash_key, -1)) % self.N
        node = hash_ring[idx][1]
        while node in self.disabled:
            if salt is None:
                node = (node + 1) % self.N
            else:
                idx = (idx + 1) % self.N
                node = hash_ring[idx][1]
        return node

    # Bounded Load
    def get_hash_node(self, key, salt=None):
        hash_ring = self._get_hash_ring(salt)
        hash_key = Hash(key, salt)
        return hash_ring[bisect.bisect_right(hash_ring, x=(hash_key, -1)) %
                         self.N][1]

    def get_boundedhash_node_list(self, key):
        n = self.get_hash_node(key)
        yield n
        while True:
            n = (n + 1) % self.N
            yield n
        # n = self.get_hash_node_churn(key); yield n
        # while True:
        #     n = (n+1)%self.N
        #     if n in self.disabled:
        #         continue
        #     yield n

    def query_boundedhash(self, key, op='mix', qid=''):
        self.stats[key] += 1
        self.cost['num_queries'] += 1
        for n in self.get_boundedhash_node_list(key):
            load = self.get_node_load(n)
            if self.capacity >= 0 and load >= self.capacity:
                continue
            else:
                return self.query_node(n, key, op=op, qid=qid)

    # Balanced Hash
    def get_balancedhash_node_list(self, key):
        yield self.get_hash_node(key)
        i = 1
        while True:
            yield self.get_hash_node(key, salt=i)
            i += 1

    def query_balancedhash(self, key, op='mix', qid=''):
        self.stats[key] += 1
        self.cost['num_queries'] += 1
        for n in self.get_balancedhash_node_list(key):
            load = self.get_node_load(n)
            if self.capacity >= 0 and load >= self.capacity:
                continue
            else:
                return self.query_node(n, key, op=op, qid=qid)

    # Hot Hash
    def get_hothash_node_legacy(self, key, qid=''):
        hash_ring = self._get_hash_ring(salt=key)
        hash_key = Hash(qid, salt=key)
        frequency = self.stats[key] / self.cost[
            'num_queries'] if key in self.stats else 0.
        if self.dyn_op:
            frequency = self.dyn_stats[key] / self.dyn_cost['num_queries']
        num_nodes = min(max(int(ceil((frequency**self.alpha) * self.N)), 1),
                        self.N)
        nodes_set = set(
            [n for _, n in self._get_hash_ring(salt=("arc", key))[:num_nodes]])
        hash_arc = [(h, n) for h, n in hash_ring if n in nodes_set]
        return hash_arc[bisect.bisect_right(hash_arc, x=(hash_key, -1)) %
                        num_nodes][1]

    def get_hothash_node(self, key, qid=''):
        # hash_ring = self._get_hash_ring(salt=key); hash_key = Hash(qid, salt=key)
        frequency = self.stats[key] / self.cost[
            'num_queries'] if key in self.stats else 0.
        if self.dyn_op:
            frequency = self.dyn_stats[key] / self.dyn_cost['num_queries']
        num_nodes = min(max(int(ceil((frequency**self.alpha) * self.N)), 1),
                        self.N)
        nodes_set = set(
            [n for _, n in self._get_hash_ring(salt=("arc", key))[:num_nodes]])
        return min(nodes_set, key=lambda n: self.get_node_load(n))
        # hash_arc = [(h,n) for h,n in hash_ring if n in nodes_set]
        # return hash_arc[bisect.bisect_right(hash_arc, x=(hash_key,-1)) % num_nodes][1]

    def get_hothash_node_churn(self, key, qid=''):
        # hash_ring = self._get_hash_ring(salt=key); hash_key = Hash(qid, salt=key)
        frequency = self.stats[key] / self.cost[
            'num_queries'] if key in self.stats else 0.
        if self.dyn_op:
            frequency = self.dyn_stats[key] / self.dyn_cost['num_queries']
        num_nodes = min(
            max(
                int(
                    ceil((frequency**self.alpha) *
                         (self.N - len(self.disabled)))), 1),
            (self.N - len(self.disabled)))
        nodes_set = set(
            [n for _, n in self._get_hash_ring(salt=("arc", key))[:num_nodes]])
        for i, node in enumerate(self.disabled):
            if node in nodes_set:
                nodes_set.remove(node)
                nodes_set.add(
                    self._get_hash_ring(salt=("arc", key))[num_nodes + i][1])
        return min(nodes_set, key=lambda n: self.get_node_load(n))
        # hash_arc = [(h,n) for h,n in hash_ring if n in nodes_set]
        # return hash_arc[bisect.bisect_right(hash_arc, x=(hash_key,-1)) % num_nodes][1]

    def get_hothash_node_vring(self, key, qid=''):
        # hash_ring = self._get_hash_ring(salt=key); hash_key = Hash(qid, salt=key)
        frequency = self.stats[key] / self.cost[
            'num_queries'] if key in self.stats else 0.
        if self.dyn_op:
            frequency = self.dyn_stats[key] / self.dyn_cost['num_queries']
        num_nodes = min(max(int(ceil((frequency**self.alpha) * self.N)), 1),
                        self.N)
        nodes_set = set([
            n for _, n in self._get_hash_ring(
                salt=("arc", self.vring_map[key]))[:num_nodes]
        ])
        return min(nodes_set, key=lambda n: self.get_node_load(n))
        # hash_arc = [(h,n) for h,n in hash_ring if n in nodes_set]
        # return hash_arc[bisect.bisect_right(hash_arc, x=(hash_key,-1)) % num_nodes][1]

    def query_hothash(self, key, op='mix', qid=''):
        query_start = time.time()
        self.stats[key] += 1
        self.cost['num_queries'] += 1
        n = self.get_hothash_node(key, qid=qid)
        self.vring_cost['time'] += (time.time() - query_start)
        return self.query_node(n, key, op=op, qid=qid)

    # rl
    def query_rl(self, key, op='mix', qid=''):
        self.stats[key] += 1
        self.cost['num_queries'] += 1
        query = Query(qid, [key], op)
        state = self.get_state(query)
        valid_actions = self.get_valid_actions(query)
        n = self.rl_agent.schedule_query(state, valid_actions)
        # print(n)
        result = self.query_node(n, key, op=op, qid=qid)
        return result

    # SPORE
    def query_sporehash(self, key, op='mix', qid='', hotkeys=[], spore_rf=1):
        self.stats[key] += 1
        self.cost['num_queries'] += 1
        nodes_set = list()
        counter = 0
        for n in self.get_balancedhash_node_list(key):
            nodes_set.append(n)
            counter += 1
            if key in hotkeys and counter < spore_rf + 1:
                continue
            else:
                break
        n = np.random.choice(nodes_set)
        return self.query_node(n, key, op=op, qid=qid)

    def query(self, key, op='mix', qid='', hotkeys=[], spore_rf=1):
        if self.hash_policy == 'hot':
            return self.query_hothash(key, op=op, qid=qid)
        elif self.hash_policy == 'rl':
            return self.query_rl(key, op=op, qid=qid)
        elif self.hash_policy == 'balanced':
            return self.query_balancedhash(key, op=op, qid=qid)
        elif self.hash_policy == 'spore':
            return self.query_sporehash(key,
                                        op=op,
                                        qid=qid,
                                        hotkeys=hotkeys,
                                        spore_rf=spore_rf)
        else:
            return self.query_boundedhash(key, op=op, qid=qid)

    def initialize_data_distribution(self):
        """Initialize data distribution across nodes using consistent hashing"""
        print("Initializing data distribution...")
        for data_item in self.data_items:
            # Use consistent hashing to determine initial placement
            hash_value = int(hashlib.md5(data_item.encode()).hexdigest(), 16)
            node_idx = hash_value % self.num_nodes
            self.query_node(n=node_idx,
                            data_id=data_item,
                            op='mix',
                            qid=f'init_cache')
        print("Data distribution initialized.")

    def get_valid_actions(self, query: Query) -> List[int]:
        """Get valid node assignments for a query"""
        valid_actions = []

        for i in range(self.num_nodes):
            valid_actions.append(i)

        return valid_actions

    def get_state(self, query: Query) -> np.ndarray:
        """Get current environment state for RL agent"""
        state = []

        node_infos = [self.get_node_info(i) for i in range(self.num_nodes)]

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
        loads = [self.get_node_load(i) for i in range(self.num_nodes)]
        max_load = max(loads) if loads and max(loads) > 0 else 1  # 避免除以0
        normalized_loads = [load / max_load for load in loads]
        return normalized_loads

    def calculate_locality_score(self, result: dict) -> float:
        """Calculate data locality score for assigning query to node"""
        if result['cache_miss'] == 1:
            locality_score = 0.0
        else:
            locality_score = 1.0

        return locality_score

    def total_size(self, obj, seen=None):
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(
                self.total_size(k, seen) + self.total_size(v, seen)
                for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(self.total_size(i, seen) for i in obj)

        return size

    @property
    def profile(self):
        infos = [self.get_node_info(n) for n in range(self.N)]
        loads = [info['num_queries'] for info in infos]
        max_load, min_load, avg_load = max(loads), min(
            loads), sum(loads) / self.N
        sigma = sqrt(sum([(l - avg_load)**2 for l in loads]) / self.N)
        imbalance = sigma * self.N / self.cost['num_queries'] if self.cost[
            'num_queries'] > 0 else 0.
        if self.dyn_op:
            loads = [value for _, value in self.node_loads.items()]
            max_load, min_load, avg_load = max(loads), min(
                loads), sum(loads) / self.N
            sigma = sqrt(sum([(l - avg_load)**2 for l in loads]) / self.N)
            imbalance = sigma * self.N / self.dyn_cost['num_queries']
        replicas = [info['num_replicas'] for info in infos]
        self.vring_cost['size'] += self.total_size(self.hash_rings)
        self.dyn_op_cost['size'] += self.total_size(self.stats)
        self.dyn_op_cost['size'] += self.total_size(self.cost)
        self.dyn_op_cost['size'] += self.total_size(self.node_loads)
        return {
            'num_nodes': self.N,
            'hash_policy': self.hash_policy,
            'num_queries': self.cost['num_queries'],
            'num_replicas': sum(replicas),
            'cache_hit_ratio': 1 - sum(replicas) / self.cost['num_queries'],
            'imbalance': imbalance,
            'max_load': max_load,
            'min_load': min_load,
            'avg_load': avg_load,
            'loads': dict(self.node_loads),
            'node_infos': infos,
            'vring_size': self.vring_cost['size'],
            'vring_time': self.vring_cost['time'],
            'dyn_op_size': self.dyn_op_cost['size'],
            'dyn_op_time': self.dyn_op_cost['time'],
        }


import sys
import numpy as np
from pyheaven import HeavenArguments, IntArgumentDescriptor, FloatArgumentDescriptor, LiteralArgumentDescriptor, SwitchArgumentDescriptor, BoolArgumentDescriptor, PrintJson


def get_lodate(num_data, B=7):
    bucket_size = num_data // B
    bucket = [[] for _ in range(B)]
    for data_id in range(num_data):
        bucket_index = min(int(data_id / bucket_size), 6)
        bucket[bucket_index].append(data_id)
    return bucket


def generate_workload(
    mode="ycsb",
    num_time_steps=10,
    num_queries_per_time_step=100,
    num_data=100,
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


def execute_workload(
    cluster,
    workload,
    time_step_duration_secs=10.0,
    spore_replication_factor=1,
    spore_threshold=1,
    node_churn=0,
):
    tot_start_time = time.time()
    results = dict()
    local_times = dict()
    futures = []

    def callback(result):
        q = result['query_id']
        results[q] = result
        if q not in local_times:
            local_times[q] = 0.0
        results[q]['local_time_start'] = local_times[q] - tot_start_time
        results[q]['local_time'] = time.time() - local_times[q]

    def query_wrapper(args):
        answer = cluster.query(**args)
        callback(answer)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for t, queries in enumerate(TQDM(workload)):
            # if t == 19:
            #     for n in range(node_churn):
            #         cluster.node_churn(n, "DISABLE")
            stat = defaultdict(int)
            for _, d, _ in queries:
                if cluster.dyn_op:
                    stat[d] += (cluster.dyn_stats[d] / cluster.dyn_bound[d])
                else:
                    stat[d] += 1
            hotkeys = sorted(stat, key=stat.get,
                             reverse=True)[:spore_threshold]
            for q, d, o in queries:
                local_times[q] = time.time()
                future = executor.submit(
                    query_wrapper, {
                        'qid': q,
                        'key': d,
                        'op': o,
                        'hotkeys': hotkeys,
                        'spore_rf': spore_replication_factor
                    })
                futures.append((q, future))
            time.sleep(time_step_duration_secs)

        for future in concurrent.futures.as_completed([f for _, f in futures]):
            pass

    tot_end_time = time.time()
    time.sleep(60)

    profile = {}
    profile['avg_latency'] = sum(r['local_time']
                                 for r in results.values()) / len(results)
    profile['total_duration_secs'] = tot_end_time - tot_start_time
    profile['avg_duration_secs'] = profile['total_duration_secs'] / len(
        results)
    profile['p99_latency'] = np.percentile(
        [r['local_time'] for r in results.values()], 99)
    profile = {**profile, **cluster.profile}
    return profile, results


if __name__ == "__main__":
    args = HeavenArguments.from_parser(descriptors=[
        IntArgumentDescriptor('N', default=20, help='Number of nodes'),
        IntArgumentDescriptor(
            'T', default=40, help='Number of query time steps'),
        IntArgumentDescriptor(
            'q', default=500, help='Average number of queries per time step'),
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
        SwitchArgumentDescriptor('d', help='debug mode'),
    ])
    workload = generate_workload(
        mode=args.M,
        num_time_steps=args.T,
        num_queries_per_time_step=args.q,
        num_data=args.D,
        skewness=args.k,
        seed=args.s,
        dyn_op=args.O,
    )
    m = sum(len(queries) for queries in workload)
    C = max(int(ceil(m / args.N * (1 + args.e))), 1)
    cluster = Cluster(
        N=args.N,
        hash_policy=args.H,
        capacity={
            'hot': -1,
            'balanced': C,
            'bounded': C,
            'cons': -1,
            'spore': -1,
            'rl': -1
        }[args.H],
        hot_alpha=args.a,
        num_data=args.D,
        random=args.R,
        dyn_op=args.O,
        debug=args.d,
    )
    if args.O and (args.H == 'bounded' or args.H == 'balanced'):
        cluster.capacity = max(
            int(ceil(cluster.dyn_cost['num_queries'] / args.N * (1 + args.e))),
            1)

    if args.H == 'rl':
        cluster.initialize_data_distribution()
        scheduler = RLQueryScheduler(
            len(cluster.get_state(Query('', [''], 'mix'))), cluster.num_nodes)
        scheduler.load_checkpoint()
        # scheduler.save_checkpoint()
        cluster.rl_agent = scheduler

    print(
        f"Executing workload with {args.N} nodes, {args.H} schedule policy"
    )
    profile = execute_workload(
        cluster,
        workload,
        time_step_duration_secs=args.t,
        spore_replication_factor=args.g,
        spore_threshold=args.r,
        node_churn=args.C,
    )
    PrintJson(profile)
