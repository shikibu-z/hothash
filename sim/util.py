import numpy as np


def with_idx(l, idxs):
    """
    Return list with elements at specified idxs
    """
    return [e for i, e in enumerate(l) if i in idxs]


def without(l, idxs):
    """
    Return list without elements at specified idxs
    """
    return [v for i, v in enumerate(l) if i not in idxs]


def consolidate(l, new_count):
    """
    Given a list and a target size return a new list of lists

    with elements dispersed evenly from original

    [1,2,3,4,5]

    size 3 [[1,2],[3],[4,5]]
    size 7 [[1], [], [2], [3], [], [4], [5]]
    """
    new_l = []

    count = len(l)
    runs = [1] * len(l)
    l_local = [e for e in l]
    if count > new_count:
        while len(runs) > new_count:
            idx = np.random.randint(len(runs) - 1)
            v = runs.pop(idx)
            runs[idx] += v
    if count < new_count:
        while len(runs) < new_count:
            idx = np.random.randint(len(runs))
            runs.insert(idx, 0)

    out = []
    for i in runs:
        run = []
        for j in range(i):
            run.append(l_local.pop())
        out.append(run)

    return out
