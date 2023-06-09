import os
import gymnasium
from gymnasium_robotics.envs.fetch import fetch_env
import numpy as np
import copy
import math
from gymnasium_robotics.utils import rotations
from gymnasium_robotics import utils
from gymnasium_robotics.envs import robot_env

# Ensure we get the path separator correct on windows
from typing import List

from envs.fetch.pick_dyn_obstacles import FetchPickDynObstaclesEnv

MODEL_XML_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'pick_dyn_obstacles.xml')

# to test max operator
class FetchPickDynObstaclesMaxEnv(FetchPickDynObstaclesEnv):

    def _compute_obstacle_rel_x_positions(self, time) -> np.ndarray:
        pos = super()._compute_obstacle_rel_x_positions(time)

        pos[1] = -0.022

        return pos

    def _sample_goal(self):
        return np.array([1.4, 0.43, 0.469])
