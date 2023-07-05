import time

from stable_baselines3 import PPO

import os
import csv

import metaworld
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from RL_PPA_monitor import RLPPAMonitor
from plots.monitor_mt10_success import review_success_rate


def train():
    models_dir = "models/teleporting_pruned/ppo_original_ratio"
    logdir = "logs/teleporting_pruned/ppo_original_ratio"
    timestamps = 8192
    number_envs_per_task = [5, 15, 30, 3, 2, 4, 3, 20, 3, 2]
    batch_size = 16384  # should be mutliple of tasks * num envs
    # We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
    # Info: (n_steps=13 and n_envs=20)
    rew_type = "rew1"

    # create env
    mt10 = metaworld.MT10()
    env_array = []
    counter = 0
    for i, (name, _) in enumerate(mt10.train_classes.items()):
        for _ in range(number_envs_per_task[i]):
            env_array.append(make_env(name, rew_type, 10, i))
            print(f"finished the {counter}nth environment")
            counter += 1
    log_name = "mt10_teleporting_pruned1"
    env_vec = SubprocVecEnv(env_array)
    env_vec = RLPPAMonitor(env_vec,
                           f"{logdir}/{log_name}",
                           multi_env=True,
                           num_tasks=10)

    # model = PPO.load("models/PPO3/538804224.zip", env=env_vec, tensorboard_log=logdir)
    if not os.path.isfile(f"{logdir}/telepotring_timer.csv"):
        # create new file
        with open(f"{logdir}/telepotring_timer.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            # create model new
            model = PPO('MlpPolicy',
                        env_vec,
                        verbose=1,
                        tensorboard_log=logdir,
                        n_steps=timestamps,
                        batch_size=batch_size, )
            iters = 0
            while True:
                print(iters)
                iters += 1
                st = time.time()
                model = model.learn(total_timesteps=timestamps,
                                    reset_num_timesteps=False,
                                    tb_log_name="PPO_teleporting_0", )
                writer.writerow([time.time() - st, iters])
                model.save(f"{models_dir}/{timestamps * iters * 13}")
                print("finished iteration {} for model {}".format(iters,
                                                                  timestamps * iters * 24))
                file.flush()
                review_success_rate(f"{logdir}/{log_name}.csv")
    else:
        # append to file
        with open(f"{logdir}/telepotring_timer.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            # load last model and continue training it
            model = PPO.load(f"{models_dir}/x.zip",
                             env=env_vec,
                             tensorboard_log=logdir)
            iters = 0
            while True:
                iters += 1
                st = time.time()
                model = model.learn(total_timesteps=timestamps,
                                    reset_num_timesteps=False,
                                    tb_log_name="PPO_0", )
                writer.writerow([time.time() - st, iters])
                model.save(f"{models_dir}/{timestamps * iters * 13}")
                print("finished iteration {} for model {}".format(iters,
                                                                  timestamps * iters * 13))
                file.flush()
                review_success_rate(f"{logdir}/{log_name}.csv")


def make_env(name, rew_type, number_of_one_hot_tasks, one_hot_task_index):
    def _init():
        return SubGoalEnv(env=name,
                          rew_type=rew_type,
                          number_of_one_hot_tasks=number_of_one_hot_tasks,
                          one_hot_task_index=one_hot_task_index)

    return _init


if __name__ == '__main__':
    train()
