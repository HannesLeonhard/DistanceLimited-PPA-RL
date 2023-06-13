# from gym.wrappers import Monitor
from typing import Tuple

import time

import numpy as np
from stable_baselines3 import SAC

import os
import csv

import metaworld
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from RL_PPA_monitor import RLPPAMonitor


def train():
    models_dir = f"models/teleporting/SAC"
    logdir = "logs/teleporting"
    timestamps = 4096  # 8192
    number_envs_per_task = [2, 3, 6, 1, 1, 1, 1, 3, 1, 1]
    batch_size = 1024  # should be multiple of tasks * num envs
    # We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
    # Info: (n_steps=13 and n_envs=20)
    rew_type = "rew1"

    # create env
    mt10 = metaworld.MT10()
    env_array = []

    for i, (name, _) in enumerate(mt10.train_classes.items()):
        for _ in range(number_envs_per_task[i]):
            env_array.append(make_env(name, rew_type, 10, i))

    env_vec = SubprocVecEnv(env_array)
    env_vec = RLPPAMonitor(env_vec,
                           "logs/teleporting/SAC/mt10_teleporting2",
                           multi_env=True,
                           num_tasks=10)

    if not os.path.isfile("logs/teleporting/SAC/telepotring_timer.csv"):
      # create new file
      with open("logs/teleporting/SAC/telepotring_timer.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        # create new model
        model = SAC('MlpPolicy',
                    env_vec,
                    verbose=1,
                    batch_size=batch_size,
                    tensorboard_log=logdir,
                    seed=13,
                    ent_coef='auto_0.5',
                    train_freq=(2, "step"))
        iters = 0
        while True:
          print(iters)
          iters += 1
          st = time.time()
          model = model.learn(total_timesteps=timestamps,
                              reset_num_timesteps=False,
                              tb_log_name="SAC_teleporting_0", )
          writer.writerow([time.time() - st, iters])
          model.save(f"{models_dir}/{timestamps * iters * 13}")
          print("finished iteration {} for model {}".format(iters,
                                                            timestamps * iters * 24))
          file.flush()
    else:
      # append to file
      with open("logs/teleporting/SAC/telepotring_timer.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        # load last model and continue training it
        model = SAC.load(f"{models_dir}/1474560.zip",
                         env=env_vec,
                         tensorboard_log=logdir)
        iters = 15
        while True:
          iters += 1
          st = time.time()
          model = model.learn(total_timesteps=timestamps,
                              reset_num_timesteps=False,
                              tb_log_name="SAC_teleporting_0", )
          writer.writerow([time.time() - st, iters])
          model.save(f"{models_dir}/{timestamps * iters * 13}")
          print("finished iteration {} for model {}".format(iters,
                                                            timestamps * iters * 13))
          file.flush()


def make_env(name, rew_type, number_of_one_hot_tasks, one_hot_task_index):

    def _init():
        return SubGoalEnv(env=name,
                          rew_type=rew_type,
                          number_of_one_hot_tasks=number_of_one_hot_tasks,
                          one_hot_task_index=one_hot_task_index)
    return _init


if __name__ == '__main__':
    train()
