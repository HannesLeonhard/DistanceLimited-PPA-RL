import numpy as np
import pandas as pd
from numpy import ndarray


def review_success_rate(file_name: str):
    try:
        raw_data = pd.read_csv(file_name, delimiter=',', low_memory=False)
        # prep the data
        raw_data['time_trunctuated'] = np.where(raw_data['time_trunctuated'] == "True", 1., 0.)
        raw_data['r'] = np.where(raw_data['r'] == '-', -20, raw_data['r'])
        raw_data['r'] = np.nan_to_num(raw_data['r'], nan=0)
        raw_data['r'] = raw_data['r'].astype(float)
        # max step size is 3 (max_it) * 15 (max node depth) * 0.01 * (step size) which is =0.45
        raw_data['distance_too_large'] = np.where(raw_data['distance_to_goal'] > 0.45, 1.0, 0.0)
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
    except:
        print("error")


def print_kpis(monitor: ndarray, values_combined: int):
    print(f"-> Success Rate {monitor[:, 5][-values_combined:].mean()}")
    print(f"-> Length {monitor[:, 1][-values_combined:].mean()}")
    print(f"-> Reward {monitor[:, 0][-values_combined:].mean()}")
