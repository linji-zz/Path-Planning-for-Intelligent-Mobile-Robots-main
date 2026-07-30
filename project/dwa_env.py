"""dwa_env.py - DQN + DWA 集成环境"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from env1 import GridEnv, OBSTACLES
from dwa_planner import DWAPlanner


def local_openness(env, target_x, target_y):
    obstacle_count = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = target_x + dx, target_y + dy
            if 0 <= nx < env.size and 0 <= ny < env.size:
                if (nx, ny) in env.obstacles:
                    obstacle_count += 1
    return 1.0 - obstacle_count / 9.0


class DWAEnv:
    def __init__(self, dwa_params=None, coupled_reward=True):
        self.grid_env = GridEnv()
        self.use_coupled = coupled_reward

        default_params = {
            'v_max': 2.0,
            'w_max': 2.0,
            'v_acc': 1.0,
            'w_acc': 1.0,
            'dt': 0.1,
            'eval_time': 1.0,
        }
        if dwa_params:
            default_params.update(dwa_params)

        self.dwa = DWAPlanner(
            v_max=default_params['v_max'],
            w_max=default_params['w_max'],
            v_acc=default_params['v_acc'],
            w_acc=default_params['w_acc'],
            dt=default_params['dt'],
            eval_time=default_params['eval_time'],
        )

        self.cont_x = 1.5
        self.cont_y = 1.5
        self.cont_theta = 0.0
        self.current_v = 0.0
        self.current_w = 0.0
        self.steps = 0
        self.done = False
        self.MAX_STEPS = 300
        self.trajectory = []

        self.size = self.grid_env.size
        self.obstacles = self.grid_env.obstacles
        self.start = self.grid_env.start
        self.goal = self.grid_env.goal
        self.actions = self.grid_env.actions
        self.action_dim = self.grid_env.action_dim
        self.agent_pos = self.grid_env.agent_pos

    def reset(self):
        self.grid_env.reset()
        self.cont_x = self.grid_env.start[0] + 0.5
        self.cont_y = self.grid_env.start[1] + 0.5
        self.cont_theta = 0.0
        self.current_v = 0.0
        self.current_w = 0.0
        self.steps = 0
        self.done = False
        self.trajectory = [(self.cont_x, self.cont_y)]
        self.agent_pos = self.grid_env.agent_pos
        return self._get_state()

    def _get_state(self):
        return self.grid_env._get_state()

    def _get_grid_pos(self):
        gx = int(round(self.cont_x))
        gy = int(round(self.cont_y))
        gx = max(0, min(self.grid_env.size - 1, gx))
        gy = max(0, min(self.grid_env.size - 1, gy))
        return (gx, gy)

    def step(self, action):
        self.steps += 1

        old_grid_pos = self._get_grid_pos()

        dx, dy = self.grid_env.actions[action]
        target_cell_x = old_grid_pos[0] + dx
        target_cell_y = old_grid_pos[1] + dy

        target_cell_x = max(0, min(self.grid_env.size - 1, target_cell_x))
        target_cell_y = max(0, min(self.grid_env.size - 1, target_cell_y))

        # FIX 1: If target cell is obstacle, stay in current cell instead of crashing
        if (target_cell_x, target_cell_y) in self.grid_env.obstacles:
            target_x = old_grid_pos[0] + 0.5
            target_y = old_grid_pos[1] + 0.5
        else:
            target_x = target_cell_x + 0.5
            target_y = target_cell_y + 0.5

        # FIX 2: Pass current theta to DWA planner so it simulates from actual heading
        v, w = self.dwa.plan(
            start_pos=(self.cont_x, self.cont_y),
            goal_pos=(target_x, target_y),
            obstacles=self.grid_env.obstacles,
            current_v=self.current_v,
            current_w=self.current_w,
            current_theta=self.cont_theta,
        )

        # FIX 3: Execute FULL DWA plan (eval_time/dt micro-steps per DQN step)
        dt = self.dwa.dt
        sub_steps = max(1, int(self.dwa.eval_time / self.dwa.dt))
        sub_dt = dt

        for s in range(sub_steps):
            nx = self.cont_x + v * math.cos(self.cont_theta) * sub_dt
            ny = self.cont_y + v * math.sin(self.cont_theta) * sub_dt
            ntheta = self.cont_theta + w * sub_dt

            check_x, check_y = int(round(nx)), int(round(ny))

            if check_x < 0 or check_x >= self.grid_env.size or check_y < 0 or check_y >= self.grid_env.size:
                self.done = True
                self.trajectory.append((nx, ny))
                self.agent_pos = self._get_grid_pos()
                return self._get_state(), -10.0, True, {'reason': 'boundary'}

            if (check_x, check_y) in self.grid_env.obstacles:
                self.done = True
                self.trajectory.append((nx, ny))
                self.agent_pos = self._get_grid_pos()
                return self._get_state(), -10.0, True, {'reason': 'collision'}

            self.cont_x, self.cont_y, self.cont_theta = nx, ny, ntheta
            self.trajectory.append((nx, ny))

            if check_x == self.grid_env.goal[0] and check_y == self.grid_env.goal[1]:
                self.done = True
                self.agent_pos = (check_x, check_y)
                self.grid_env.agent_pos = self.agent_pos
                return self._get_state(), 50.0, True, {'reason': 'goal'}

        self.current_v = v
        self.current_w = w
        new_grid_pos = self._get_grid_pos()
        self.agent_pos = new_grid_pos
        self.grid_env.agent_pos = new_grid_pos

        old_dist = math.hypot(old_grid_pos[0] - self.grid_env.goal[0], old_grid_pos[1] - self.grid_env.goal[1])
        new_dist = math.hypot(new_grid_pos[0] - self.grid_env.goal[0], new_grid_pos[1] - self.grid_env.goal[1])

        if new_grid_pos == self.grid_env.goal:
            self.done = True
            return self._get_state(), 50.0, True, {'reason': 'goal'}

        if self.steps >= self.MAX_STEPS:
            self.done = True
            reward = (old_dist - new_dist) * 2.0
            return self._get_state(), reward, True, {'reason': 'max_steps'}
        reward_dist = (old_dist - new_dist) * 2.0

        if self.use_coupled:
            angle_to_subgoal = math.atan2(target_y - self.cont_y, target_x - self.cont_x)
            alignment_reward = 0.5 * math.cos(self.cont_theta - angle_to_subgoal)
            openness = local_openness(self.grid_env, new_grid_pos[0], new_grid_pos[1])
            openness_reward = 0.3 * openness
            reward = reward_dist + alignment_reward + openness_reward
        else:
            reward = reward_dist

        return self._get_state(), reward, False, {}

    def render(self, path=None, ax=None, title=""):
        if path is None and self.trajectory:
            path = self.trajectory
        elif path is None:
            path = []
        self.grid_env.render(path=path, ax=ax, title=title)

    def show_map(self):
        self.grid_env.show_map()


if __name__ == "__main__":
    env = DWAEnv()
    state = env.reset()
    print("DWAEnv")
    print(f"State dim: {len(state)}, start: {env.start}, goal: {env.goal}")
    print(f"Continuous pos: ({env.cont_x:.2f}, {env.cont_y:.2f})")

    next_state, reward, done, info = env.step(0)
    print(f"After step: pos=({env.cont_x:.2f}, {env.cont_y:.2f})")
    print(f"Grid={env.agent_pos}, reward={reward:.2f}, done={done}")
