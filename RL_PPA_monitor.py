import time
import warnings
from typing import Optional, Tuple
import os
import csv

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


# Compied from Vector Monitor and changed
class RLPPAMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.
    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.
    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
            self,
            venv: VecEnv,
            filename: Optional[str] = None,
            info_keywords: Tuple[str, ...] = (),
            multi_env=False,
            num_tasks=1,
    ):
        self.num_tasks = num_tasks
        self.multi_env = multi_env
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_returns = None
        self.episode_lengths = None
        self.episode_count = 0
        self.t_start = time.time()
        self.filename = filename
        self.total_time_subgoal = 0

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        if not filename:
            self.results_writer = None

            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": env_id},
                extra_keys=("time_step",
                            "taskid",
                            "success",
                            "time_ppa_ns",
                            "no_A_path_times",
                            "time_trunctuated",
                            "distance_to_goal",
                            "initial_pos",
                            "subgoal_pos",
                            'action_is_goal_pos',
                            'time_in_mujoco',
                            'total_time_subgoal',
                            'time_subgoalstep_s')
            )
        elif not os.path.isfile(filename):
            with open(filename + ".csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["r",
                                 "l",
                                 "t",
                                 "time_step",
                                 "taskid",
                                 "success",
                                 "time_ppa_ns",
                                 "no_A_path_times",
                                 "time_trunctuated",
                                 "distance_to_goal",
                                 "initial_pos",
                                 "subgoal_pos",
                                 "action_is_goal_pos",
                                 'time_in_mujoco',
                                 'total_time_subgoal',
                                 'time_subgoalstep_s'])
            # create new file
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": env_id},
                extra_keys=("time_step",
                            "taskid",
                            "success",
                            "time_ppa_ns",
                            "no_A_path_times",
                            "time_trunctuated",
                            "distance_to_goal",
                            "initial_pos",
                            "subgoal_pos",
                            "action_is_goal_pos",
                            'time_in_mujoco',
                            'total_time_subgoal',
                            'time_subgoalstep_s')
            )
        else:
            # append to existing logging file
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": env_id},
                extra_keys=("time_step",
                            "taskid",
                            "success",
                            "time_ppa_ns",
                            "no_A_path_times",
                            "time_trunctuated",
                            "distance_to_goal",
                            "initial_pos",
                            "subgoal_pos",
                            "action_is_goal_pos",
                            'time_in_mujoco',
                            'total_time_subgoal',
                            'time_subgoalstep_s'),
            )
        self.info_keywords = info_keywords
        # added:
        self.successes = [0] * venv.num_envs

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        # set up start time to calculate how much time we spend in self.venv.step_wait()
        st = time.time()
        obs, rewards, dones, infos = self.venv.step_wait()
        et = time.time()
        self.total_time_subgoal += et - st
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])

        for i in range(len(dones)):
            # to give success metric
            if infos[i]["success"]:
                self.successes[i] = 1
            # add when done
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                if self.multi_env:
                    task_id = onehot_to_task_id(obs[i][-self.num_tasks:])
                    episode_info = {"r": episode_return,
                                    "l": episode_length,
                                    "t": round(time.time() - self.t_start, 6),
                                    "time_step": et - st,
                                    "taskid": task_id,
                                    "success": self.successes[i],
                                    "time_ppa_ns": info['time_in_ppa'],
                                    "no_A_path_times": info['number_no_A_path'],
                                    "time_trunctuated": info["TimeLimit.truncated"],
                                    "distance_to_goal": info['distance_to_goal'],
                                    "initial_pos": info['initial_pos'],
                                    "subgoal_pos": info['subgoal_pos'],
                                    "action_is_goal_pos": info['action_is_goal_pos'],
                                    'time_in_mujoco': info['time_in_mujoco'],
                                    'total_time_subgoal': self.total_time_subgoal,
                                    'time_subgoalstep_s': info['time_subgoalstep_s']}
                    with open(self.filename + ".csv", mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            episode_return,
                            episode_length,
                            round(time.time() - self.t_start, 6),
                            et - st,
                            task_id,
                            self.successes[i],
                            info['time_in_ppa'],
                            info['number_no_A_path'],
                            info["TimeLimit.truncated"],
                            info['distance_to_goal'],
                            info['initial_pos'],
                            info['subgoal_pos'],
                            info['action_is_goal_pos'],
                            info['time_in_mujoco'],
                            self.total_time_subgoal,
                            info['time_subgoalstep_s']
                        ])
                        file.flush()
                else:
                    episode_info = {"r": episode_return,
                                    "l": episode_length,
                                    "t": round(time.time() - self.t_start, 6),
                                    "time_step": et - st,
                                    "success": self.successes[i],
                                    "time_ppa_ns": info['time_in_ppa'],
                                    "no_A_path_times": info['number_no_A_path'],
                                    "time_trunctuated": info["TimeLimit.truncated"],
                                    "distance_to_goal": info['distance_to_goal'],
                                    "initial_pos": info['initial_pos'],
                                    "subgoal_pos": info['subgoal_pos'],
                                    "action_is_goal_pos": info['action_is_goal_pos'],
                                    'time_in_mujoco': info['time_in_mujoco'],
                                    'total_time_subgoal': self.total_time_subgoal,
                                    'time_subgoalstep_s': info['time_subgoalstep_s']}
                    with open(self.filename + ".csv", mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            episode_return,
                            episode_length,
                            round(time.time() - self.t_start, 6),
                            et - st,
                            self.successes[i],
                            info['time_in_ppa'],
                            info['number_no_A_path'],
                            info["TimeLimit.truncated"],
                            info['distance_to_goal'],
                            info['initial_pos'],
                            info['subgoal_pos'],
                            info['action_is_goal_pos'],
                            info['time_in_mujoco'],
                            self.total_time_subgoal,
                            info['time_subgoalstep_s']
                        ])
                        file.flush()
                # print(info)
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
                # if success is logged in environment, then set 0 again
                self.successes[i] = 0
            else:
                info = infos[i].copy()
                task_id = onehot_to_task_id(obs[i][-self.num_tasks:])
                # print time info
                if self.multi_env:
                    with open(self.filename + ".csv", mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            0,
                            0,
                            round(time.time() - self.t_start, 6),
                            et - st,
                            task_id,
                            -1,
                            info['time_in_ppa'],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            info['time_in_mujoco'],
                            self.total_time_subgoal,
                            info['time_subgoalstep_s']
                        ])
                        file.flush()
                else:
                    with open(self.filename + ".csv", mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            0,
                            0,
                            round(time.time() - self.t_start, 6),
                            et - st,
                            -1,
                            info['time_in_ppa'],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            info['time_in_mujoco'],
                            self.total_time_subgoal,
                            info['time_subgoalstep_s']
                        ])
                        file.flush()
        return obs, rewards, dones, new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()


def onehot_to_task_id(nohup) -> int:
    for i in range(len(nohup)):
        if nohup[i] == 1:
            return i
    raise Exception("not a currect one-hot in: nohup_to_task_id")
