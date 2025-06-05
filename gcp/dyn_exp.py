import time
import json
import bisect
import hashlib
import requests
import threading
import concurrent.futures
from math import ceil, sqrt
from collections import defaultdict
from pyheaven import TQDM

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def Hash(key, salt=None):
    return int(hashlib.md5((f"{key}" if salt is None else f"{key}|{salt}").encode('utf-8')).hexdigest(), 16)

class Cluster:
    def __init__(self, N=20, hash_policy='hot', capacity=-1, hot_alpha=1.0, debug=False):
        self.config = json.load(open('config.json'))
        self.hash_policy = hash_policy
        self.capacity = capacity
        self.alpha = hot_alpha
        self.debug = debug
        self.init(N)

    def get_node_api(self, n):
        node_internal_ip = self.config['node_internal_ips'][f'hothash-node-{n+1:02d}']; return f"http://{node_internal_ip}:8080"

    def query_node(self, n, data_id, op='mix', qid=''):
        try:
            self.node_loads[n] += 1
            if self.debug:
                return {"result": 0, 'tot_server_time': 0, 'fetch_time' : 0, 'compute_time': 0, 'data_id': 0, 'node_id': 0, 'query_id': 0, 'query_op': None}
            return requests.get(self.get_node_api(n)+'/query', params={'op': op, 'data_id': data_id, 'query_id': qid}).json()
        except Exception as e:
            print(n, data_id, e); return None

    def clear_node(self, n):
        try:
            self.node_loads[n] = 0
            if self.debug:
                return
            requests.get(self.get_node_api(n)+'/clear_cache')
        except Exception as e:
            print(n, e)

    def get_node_info(self, n):
        try:
            if self.debug:
                return {"version": 0, "num_queries": 0, "num_replicas": 0, "cache_size": 0, "cached_keys": list()}
            return requests.get(self.get_node_api(n)+'/info').json()
        except Exception as e:
            print(n, e); return None

    def init(self, N):
        self.N = N
        self.cost = {
            'num_queries': 0,
            'cache_hit': 0,
            'replica': 0
        }
        self.hash_rings = {}
        self.node_loads = defaultdict(int)
        self.stats = defaultdict(int)
        for n in range(N):
            self.clear_node(n)

    def get_node_load(self, n):
        return self.node_loads[n]
        # return self.get_node_info(n)['num_queries']

    def _get_hash_ring(self, salt=None):
        if salt in self.hash_rings: return self.hash_rings[salt]
        self.hash_rings[salt] = sorted([(Hash(i, salt), i) for i in range(self.N)])
        return self.hash_rings[salt]

    # Bounded Load
    def get_hash_node(self, key, salt=None):
        hash_ring = self._get_hash_ring(salt); hash_key = Hash(key, salt)
        return hash_ring[bisect.bisect_right(hash_ring, x=(hash_key,-1)) % self.N][1]

    def get_boundedhash_node_list(self, key):
        n = self.get_hash_node(key); yield n
        while True:
            n = (n+1)%self.N; yield n

    def query_boundedhash(self, key, op='mix', qid=''):
        self.stats[key] += 1; self.cost['num_queries'] += 1
        for n in self.get_boundedhash_node_list(key):
            load = self.get_node_load(n)
            if self.capacity>=0 and load>=self.capacity:
                continue
            else:
                return self.query_node(n, key, op=op, qid=qid)

    # Balanced Hash
    def get_balancedhash_node_list(self, key):
        yield self.get_hash_node(key); i = 1
        while True:
            yield self.get_hash_node(key, salt=i); i += 1

    def query_balancedhash(self, key, op='mix', qid=''):
        self.stats[key] += 1; self.cost['num_queries'] += 1
        for n in self.get_balancedhash_node_list(key):
            load = self.get_node_load(n)
            if self.capacity>=0 and load>=self.capacity:
                continue
            else:
                return self.query_node(n, key, op=op, qid=qid)

    # Hot Hash
    def get_hothash_node_legacy(self, key, qid=''):
        hash_ring = self._get_hash_ring(salt=key); hash_key = Hash(qid, salt=key)
        frequency = self.stats[key]/self.cost['num_queries'] if key in self.stats else 0.
        num_nodes = min(max(int(ceil((frequency**self.alpha)*self.N)), 1), self.N)
        nodes_set = set([n for h,n in self._get_hash_ring(salt=("arc",key))[:num_nodes]])
        hash_arc = [(h,n) for h,n in hash_ring if n in nodes_set]
        return hash_arc[bisect.bisect_right(hash_arc, x=(hash_key,-1)) % num_nodes][1]

    def get_hothash_node(self, key, qid=''):
        hash_ring = self._get_hash_ring(salt=key); hash_key = Hash(qid, salt=key)
        frequency = self.stats[key]/self.cost['num_queries'] if key in self.stats else 0.
        num_nodes = min(max(int(ceil((frequency**self.alpha)*self.N)), 1), self.N)
        nodes_set = set([n for h,n in self._get_hash_ring(salt=("arc",key))[:num_nodes]])
        return min(nodes_set, key=lambda n:self.get_node_load(n))
        # hash_arc = [(h,n) for h,n in hash_ring if n in nodes_set]
        # return hash_arc[bisect.bisect_right(hash_arc, x=(hash_key,-1)) % num_nodes][1]

    def query_hothash(self, key, op='mix', qid=''):
        self.stats[key] += 1; self.cost['num_queries'] += 1
        n = self.get_hothash_node(key, qid=qid)
        return self.query_node(n, key, op=op, qid=qid)

    def query(self, key, op='mix', qid=''):
        if self.hash_policy == 'hot':
            return self.query_hothash(key, op=op, qid=qid)
        elif self.hash_policy == 'balanced':
            return self.query_balancedhash(key, op=op, qid=qid)
        else:
            return self.query_boundedhash(key, op=op, qid=qid)

    @property
    def profile(self):
        infos = [self.get_node_info(n) for n in range(self.N)]
        loads = [info['num_queries'] for info in infos]
        max_load, min_load, avg_load = max(loads), min(loads), sum(loads)/self.N
        sigma = sqrt(sum([(l-avg_load)**2 for l in loads])/self.N)
        imbalance = sigma*self.N/self.cost['num_queries'] if self.cost['num_queries']>0 else 0.
        replicas = [info['num_replicas'] for info in infos]
        return {
            'num_nodes': self.N,
            'hash_policy': self.hash_policy,
            'num_queries': self.cost['num_queries'],
            'num_replicas': sum(replicas),
            'cache_hit_ratio': 1-sum(replicas)/self.cost['num_queries'],
            'imbalance': imbalance,
            'max_load': max_load,
            'min_load': min_load,
            'avg_load': avg_load,
            'loads': dict(self.node_loads),
            'node_infos': infos
        }

import sys
import numpy as np
from pyheaven import HeavenArguments, IntArgumentDescriptor, FloatArgumentDescriptor, LiteralArgumentDescriptor, SwitchArgumentDescriptor, PrintJson

def get_lodate(num_data, B=7):
    bucket_size = num_data // B
    bucket = [[] for _ in range(B)]
    for data_id in range(num_data):
        bucket_index = min(int(data_id / bucket_size), 6)
        bucket[bucket_index].append(data_id)
    return bucket

def generate_workload(
    mode = "ycsb",
    num_time_steps = 10,
    num_queries_per_time_step = 100,
    num_data = 100,
    skewness = 1.01,
    seed = 42,
):
    np.random.seed(seed)
    data = list()
    query_id = 0
    lodate = get_lodate(num_data, B=7)
    N = num_queries_per_time_step * num_time_steps
    time_steps = [0 for _ in range(num_time_steps)]
    for t in range(N):
        time_steps[np.random.randint(num_time_steps)] += 1
    for t in range(num_time_steps):
        num_queries = time_steps[t]
        data_time_step = list()
        if mode == 'uniform':
            data_ids = np.random.choice(num_data, num_queries, replace=True)
        elif mode == "ycsb":
            data_ids = [Hash(x)%num_data for x in np.random.zipf(skewness, num_queries)]
        elif mode == "ssb":
            bucket_ids = [Hash(x)%7 for x in np.random.zipf(skewness, num_queries)]
            all_data_ids = [np.random.choice(lodate[bucket_id]) for bucket_id in bucket_ids for _ in range(max(ceil(0.2 * len(lodate[bucket_id])), 1))]
            data_ids = np.random.choice(all_data_ids, num_queries, replace=False)
        for i in range(num_queries):
            data_time_step.append((f"Q{query_id:06d}", f"D{data_ids[i]:03d}", 'mix')); query_id += 1
        data.append(data_time_step)

    # dynamic query operation
    ops = ['sum', 'mean', 'min', 'max']
    keys = [f"D{d:03d}" for d in range(num_data)]
    key2op = dict()
    for key in keys:
        key2op[key] = np.random.choice(ops)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j][0], data[i][j][1], key2op[data[i][j][1]])

    return data

def execute_workload(
    cluster, workload,
    time_step_duration_secs = 10.0,
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
        results[q]['local_time_start'] = local_times[q]-tot_start_time
        results[q]['local_time'] = time.time()-local_times[q]

    def query_wrapper(args):
        answer = cluster.query(**args)
        callback(answer)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for t, queries in enumerate(TQDM(workload)):
            for q, d, o in queries:
                local_times[q] = time.time()
                future = executor.submit(query_wrapper, {'qid': q, 'key': d, 'op': o})
                futures.append((q, future))
            time.sleep(time_step_duration_secs)

        for future in concurrent.futures.as_completed([f for _, f in futures]):
            pass

    tot_end_time = time.time()
    time.sleep(60)

    profile = {}
    profile['avg_latency'] = sum(r['local_time'] for r in results.values())/len(results)
    profile['total_duration_secs'] = tot_end_time - tot_start_time
    profile['avg_duration_secs'] = profile['total_duration_secs']/len(results)
    profile['p99_latency'] = np.percentile([r['local_time'] for r in results.values()], 99)
    profile = {**profile, **cluster.profile}
    return profile, results

if __name__=="__main__":
    args = HeavenArguments.from_parser(descriptors=[
        IntArgumentDescriptor    ('N', default=       20, help='Number of nodes'),
        IntArgumentDescriptor    ('T', default=       40, help='Number of query time steps'),
        IntArgumentDescriptor    ('q', default=      500, help='Average number of queries per time step'),
        IntArgumentDescriptor    ('D', default=       15, help='Number of data items'),
        LiteralArgumentDescriptor('M', default=   'ycsb', choices=['uniform', 'ycsb', 'ssb'], help='Workload mode'),
        FloatArgumentDescriptor  ('k', default=      1.3, help="Skewness parameter for Zipf's law"),
        IntArgumentDescriptor    ('s', default=       42, help='Random seed'),

        FloatArgumentDescriptor  ('t', default=  30.0, help='Duration of each time step in seconds'),
        LiteralArgumentDescriptor('H', default= 'hot', choices=['hot', 'balanced', 'bounded', 'cons'], help='Hash policy'),
        FloatArgumentDescriptor  ('a', default=  1.00, help='Hot alpha'),
        FloatArgumentDescriptor  ('e', default= 0.300, help='Balanced epsilon'),

        SwitchArgumentDescriptor ('d', help='debug mode'),
    ])
    workload = generate_workload(
        mode = args.M,
        num_time_steps = args.T,
        num_queries_per_time_step = args.q,
        num_data = args.D,
        skewness = args.k,
        seed = args.s,
    )
    m = sum(len(queries) for queries in workload)
    C = max(int(ceil(m/args.N*(1+args.e))),1)
    cluster = Cluster(
        N = args.N,
        hash_policy = args.H,
        capacity = {
            'hot': -1,
            'balanced': C,
            'bounded': C,
            'cons': -1
        }[args.H],
        hot_alpha = args.a,
        debug = args.d,
    )
    profile = execute_workload(cluster, workload,
        time_step_duration_secs = args.t,
    )
    PrintJson(profile)