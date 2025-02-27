from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import os
import pandas as pd
import streamlit as st
import numpy as np
import math
import statistics as stat

from entities import Table, QueryPlan, QueryOp, PerformaceDeltas, Fragment
from policy import DefaultPolicy
from workload import AbcWorkload
from controller import (
    SimController,
    SimControllerConfig,
    AbcPerformanceModel,
    DefaultPerformanceModel,
)
from dataset import (
    generate_table,
    generate_queryop,
    generate_workload,
)
from utils import trial_exists, load_trial, save_trial, mpl_line_plot

from entities import ConsistentHashing

TRIAL_ID = "CHBL"
NODE_COUNT = 20
STEPS = 10000


class TestWorkload(AbcWorkload):
    """
    Workload specifies both the dataset specification and provides a means to
    generate queries on the specified dataset over time.
    """

    def __init__(self, tables: List[Table]):
        self.tables = tables

    def get_work(self, timestep) -> List[QueryPlan]:
        if timestep <= 400:
            workload = generate_workload(self.tables, "zipfian", 750, 10)
        else:
            workload = list()
        return workload


# Define query scheduling policy
class TestPolicy(DefaultPolicy):
    def set_workload(self, workload):
        self.workload = workload
        for column in self.workload.tables[0].columns[0:2]:
            for fragment in column.fragments:
                self.cache_set.add(fragment)

    def manage_tasks(self, timestep, nodes, idle_tasks, running_tasks, ch_ring):
        idle_tasks = idle_tasks.copy()
        assignments = ch_ring.assign_task_chbl(idle_tasks, 0.3)
        return assignments

    def should_cache(self, node, entity) -> bool:
        return True

    def manage_cache(self, timestep, nodes, idle_tasks, running_tasks):
        # we only want to cache on node 0
        self.nodes = nodes
        return self.should_cache


task_logs = []
cache_hit = []
task_hist = []


def run_trial(trial_id):
    config = SimControllerConfig(compute_node_count=NODE_COUNT, virtual_node_count=1)

    dataset = generate_table(1, 1, 512 * 16, 512, 32)
    workload = TestWorkload(dataset)

    # Use default policy, no-caching, uniform task/node allocation
    policy = TestPolicy([])
    # policy.set_workload(workload)

    performance_model = DefaultPerformanceModel(config)
    controller = SimController(config, workload, policy, performance_model)

    for i in tqdm(range(STEPS)):
        controller.step()

    cluster_df = pd.DataFrame.from_records(controller.cluster_logs)
    node_df = pd.DataFrame.from_records(controller.node_logs)

    query_times = controller.get_query_times()

    global task_logs, cache_hit, task_hist
    task_logs = controller.task_logs
    cache_hit = controller.get_cache_hit()
    task_hist = controller.get_task_hist()

    # save_trial(TRIAL_ID, (cluster_df, node_df, query_times))
    return cluster_df, node_df, query_times


cluster_df, node_df, query_times = run_trial(TRIAL_ID)
# if not trial_exists(TRIAL_ID):
#     run_trial(TRIAL_ID)

columns = st.columns(2)
node_dfs = [node_df[node_df["node"] == i] for i in range(NODE_COUNT)]

print(len(query_times))
print(np.mean(query_times) / 1000)
print(np.percentile(np.array(query_times) / 1000, 99))
