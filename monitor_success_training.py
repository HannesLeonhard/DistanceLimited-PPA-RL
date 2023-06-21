import numpy as np
import pandas as pd
from numpy import ndarray

from plots.plot_pick_place import

def review_success_rate(file_name: str):
    raw_data = pd.read_csv(file_name, delimiter=',', low_memory=False)
    # prep the data
    raw_data['time_trunctuated'] = np.where(raw_data['time_trunctuated'] == "True", 1., 0.)
    raw_data['r'] = np.where(raw_data['r'] == '-', -20, raw_data['r'])
    raw_data['r'] = raw_data['r'].astype(float)
    # max step size is 3 (max_it) * 15 (max node depth) * 0.01 * (step size) which is =0.45
    raw_data['distance_too_large'] = np.where(raw_data['distance_to_goal'] > 0.45, 1.0, 0.0)
    print(raw_data.drop(['initial_pos', 'subgoal_pos'], axis=1).head())
    monitor: ndarray = np.asarray(raw_data.drop(['initial_pos', 'subgoal_pos'], axis=1), dtype=np.float32)
    for index, title in enumerate(
            ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2',
             'button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2']):
        filter_monitor = monitor[:, 4] == index
        filtered_monitor = monitor[filter_monitor]
        print(5 * "-" + f"Task {title}" + 5 * "-")
        print_kpis(filtered_monitor, 1000)
    # free memory up
    del raw_data, monitor


def print_kpis(monitor: ndarray, values_combined: int):
    x_time_steps_values = monitor[:, 1]
    size = len(x_time_steps_values) - (len(x_time_steps_values) % values_combined)
    x_time_steps_values = x_time_steps_values[:size]
    x_time_steps_values = x_time_steps_values.reshape(
        (int(len(x_time_steps_values) / values_combined), values_combined))
    x_time_steps_values = np.sum(x_time_steps_values, axis=1)
    x_intervals = [0]
    for j in x_time_steps_values:
        new_time_step = x_intervals[-1] + j
        x_intervals.append(new_time_step)
    x_intervals = np.array(x_intervals)
    # success rate:
    successes = monitor[:, 5]
    successes = successes[:size]
    successes = successes.reshape((int(len(successes) / values_combined), values_combined))
    y_success_values = np.average(successes, axis=1)
    y_success_values = np.insert(y_success_values, 0, 0)

    # delete some values to make it more spiky
    for _ in range(3):
        x_intervals = np.delete(x_intervals, np.arange(0, x_intervals.size, 2))
        y_success_values = np.delete(y_success_values, np.arange(0, y_success_values.size, 2))
    y_success_values = y_success_values * 100
    print(f"-> Success Rate {y_success_values[-1]}")
    print(f"-> Length {monitor[:, 1][-values_combined:].mean()}")
    print(f"-> Reward {monitor[:, 0][-values_combined:].mean()}")
