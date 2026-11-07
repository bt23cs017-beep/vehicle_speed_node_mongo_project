import sys
import types

import numpy as np


def _linear_sum_assignment(cost_matrix):
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")

    row_count, column_count = costs.shape
    if row_count == 0 or column_count == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    transposed = row_count > column_count
    if transposed:
        costs = costs.T

    row_count, column_count = costs.shape
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    assigned_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        assigned_row[0] = row
        current_column = 0
        minimum_cost = [float("inf")] * (column_count + 1)
        used_column = [False] * (column_count + 1)

        while True:
            used_column[current_column] = True
            current_row = assigned_row[current_column]
            delta = float("inf")
            next_column = 0

            for column in range(1, column_count + 1):
                if used_column[column]:
                    continue

                reduced_cost = (
                    costs[current_row - 1, column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum_cost[column]:
                    minimum_cost[column] = reduced_cost
                    previous_column[column] = current_column
                if minimum_cost[column] < delta:
                    delta = minimum_cost[column]
                    next_column = column

            for column in range(column_count + 1):
                if used_column[column]:
                    row_potential[assigned_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum_cost[column] -= delta

            current_column = next_column
            if assigned_row[current_column] == 0:
                break

        while True:
            next_column = previous_column[current_column]
            assigned_row[current_column] = assigned_row[next_column]
            current_column = next_column
            if current_column == 0:
                break

    pairs = sorted(
        (assigned_row[column] - 1, column - 1)
        for column in range(1, column_count + 1)
        if assigned_row[column]
    )
    rows = np.array([pair[0] for pair in pairs], dtype=int)
    columns = np.array([pair[1] for pair in pairs], dtype=int)
    if transposed:
        rows, columns = columns, rows
        order = np.argsort(rows)
        rows, columns = rows[order], columns[order]
    return rows, columns


try:
    from scipy.optimize import linear_sum_assignment
except (ImportError, OSError) as error:
    sys.modules.pop("scipy.optimize", None)
    scipy_optimize = types.ModuleType("scipy.optimize")
    scipy_optimize.linear_sum_assignment = _linear_sum_assignment
    sys.modules["scipy.optimize"] = scipy_optimize
    print(f"SciPy assignment unavailable; using Python fallback: {error}")

from deep_sort_realtime.deepsort_tracker import DeepSort