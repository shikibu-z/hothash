import mmh3
import numpy as np
from math import ceil
from typing import List

from entities import Table, Column, Fragment, QueryOp, QueryPlan


def generate_table(num_tables, num_columns, total_size, fragment_size, chunk_size):
    # generate simulation table
    num_fragments = ceil(total_size / fragment_size)
    # generate all fragments
    fragments = [Fragment(i, fragment_size, chunk_size) for i in range(num_fragments)]

    # assign fragments to columns
    col_fragments = [None] * num_columns
    while len(fragments) > 0:
        for i in range(num_columns):
            if col_fragments[i] == None:
                col_fragments[i] = []
            if len(fragments):
                col_fragments[i].append(fragments.pop())

    # generate all columns
    columns = []
    for i in range(len(col_fragments)):
        columns.append(Column(f"col_{i}", col_fragments[i]))

    # assign columns to tables
    table_cols = [None] * num_tables
    while len(columns) > 0:
        for i in range(num_tables):
            if table_cols[i] == None:
                table_cols[i] = []
            table_cols[i].append(columns.pop())

    # generate all tables
    tables = []
    for i in range(len(table_cols)):
        tables.append(Table(f"table_{i}", table_cols[i]))

    return tables


def generate_queryop(tables, mode):
    num_tables = len(tables)
    num_columns = len(tables[0].columns) * num_tables
    num_fragments = len(tables[0].columns[0].fragments) * num_columns

    # determine fragment id by distribution
    fragment_id = None
    if mode == "zipfian":
        zipf_sample = str(np.random.zipf(1.3))
        fragment_id = mmh3.hash(zipf_sample, signed=False) % num_fragments
    else:
        fragment_id = np.random.randint(0, num_fragments)

    table_id = None
    column_id = None
    for table in tables:
        for column in table.columns:
            fids = [fragment.id for fragment in column.fragments]
            if fragment_id in fids:
                table_id = int(table.name.split("_")[-1])
                column_id = table.columns.index(column)
                fragment_id = fids.index(fragment_id)
                break
        if table_id != None:
            break

    # determine fragments to read by uniform distribution
    fragment_range = None
    if fragment_id + 1 < len(tables[0].columns[0].fragments):
        fragment_range = np.random.randint(
            fragment_id + 1, len(tables[0].columns[0].fragments)
        )

    # generate operator object and return
    operator = QueryOp(
        "scan",
        [tables[table_id].columns[column_id].fragments[fragment_id]],
        # tables[table_id].columns[column_id].fragments[fragment_id:fragment_range],
        parent=None,
        deps=None,
    )

    return operator


def generate_workload(dataset, mode, max_query, max_operation):
    # generate workload for each time step
    queries = []

    for i in range(50):
        operations = []
        for j in range(1):
            operation = generate_queryop(dataset, mode)
            operations.append(operation)

        query = QueryPlan(operations, operations[0])
        queries.append(query)

    return queries
