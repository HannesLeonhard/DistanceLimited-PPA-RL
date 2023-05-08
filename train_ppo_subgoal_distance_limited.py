from stable_baselines3 import PPO
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from RL_PPA_monitor import RLPPAMonitor

import DistanceLimitedActorCriticPolicy

import time
import csv
import os

def train():
    models_dir = f"models/PPO"
    logdir = "logs"
    TIMESTEPS = 256
    print("started at {}".format(time.localtime(time.time())))
    env = SubGoalEnv("pick-place-v2", rew_type="rew1")
    env_vec = SubprocVecEnv([lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             #lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env,  # lambda: env,
                             ])
    env_vec = RLPPAMonitor(env_vec, "logs/supgoal_analytics_logs_monitor.csv", )

    if not os.path.isfile("logs/obstacle_timer.csv"):
      # create new file
      with open("logs/obstacle_timer.csv",mode="w",newline="") as file:
        writer = csv.writer(file)
        # create model new
        model = PPO(DistanceLimitedActorCriticPolicy,
                env_vec, verbose=1, 
                tensorboard_log=logdir, 
                n_steps=TIMESTEPS,
                batch_size=4096, )
        iters = 0
        while True:
          print(iters)
          iters += 1
          st = time.time()
          model = model.learn(total_timesteps=TIMESTEPS, 
                              reset_num_timesteps=False,
                              tb_log_name="PPO_0", )
          writer.writerow([time.time() - st, iters])
          model.save(f"{models_dir}/{TIMESTEPS * iters * 24}")
          print("finished iteration {} for model {}".format(iters, 
                                                            TIMESTEPS * iters * 24))
    else:
      # append to file
      with open("logs/obstacle_timer.csv",mode="a",newline="") as file:
        writer = csv.writer(file)
        # load last model and continue training it
        model = PPO.load("models/PPO/2887680.zip", 
                          env=env_vec, 
                          tensorboard_log=logdir)
        iters = 470
        while True:
          iters += 1
          st = time.time()
          model = model.learn(total_timesteps=TIMESTEPS, 
                              reset_num_timesteps=False,
                              tb_log_name="PPO_0", )
          writer.writerow([time.time() - st, iters])
          model.save(f"{models_dir}/{TIMESTEPS * iters * 24}")
          print("finished iteration {} for model {}".format(iters, 
                                                            TIMESTEPS * iters * 24))

if __name__ == '__main__':
    train()
