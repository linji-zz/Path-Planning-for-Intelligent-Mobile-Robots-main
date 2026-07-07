import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

GRID_SIZE = 15  # 从25缩小到15
ACTION_DIM = 8

class GridEnv:
    def __init__(self, obstacle_ratio=0.10, seed=None):
        self.size = GRID_SIZE
        self.actions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.action_dim = len(self.actions)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.obstacles = self._generate_obstacles(obstacle_ratio)
        self.start = (1, 1)
        self.goal = (GRID_SIZE-2, GRID_SIZE-2)  # 终点在(13,13)
        while self.start in self.obstacles:
            self.start = (random.randint(1, self.size-2), random.randint(1, self.size-2))
        while self.goal in self.obstacles or self.goal == self.start:
            self.goal = (random.randint(1, self.size-2), random.randint(1, self.size-2))
        self.reset()

    def _generate_obstacles(self, ratio):
        obstacles = set()
        num_obs = int(self.size * self.size * ratio)
        attempts = 0
        while len(obstacles) < num_obs and attempts < 10000:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if (x,y) not in [(1,1), (self.size-2, self.size-2)]:
                obstacles.add((x,y))
            attempts += 1
        return obstacles

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        x, y = self.agent_pos
        state[0, x, y] = 1.0
        for (ox, oy) in self.obstacles:
            state[1, ox, oy] = 1.0
        gx, gy = self.goal
        state[2, gx, gy] = 1.0
        return state

    def step(self, action):
        self.steps += 1
        dx, dy = self.actions[action]
        old_pos = self.agent_pos
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy

        # 边界检查
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            reward = -10.0
            self.done = True
            return self._get_state(), reward, True, {'reason': 'boundary'}

        new_pos = (new_x, new_y)
        # 障碍物检查
        if new_pos in self.obstacles:
            reward = -10.0
            self.done = True
            return self._get_state(), reward, True, {'reason': 'collision'}

        # 执行移动
        self.agent_pos = new_pos

        # ===== 稠密奖励计算 =====
        old_dist = math.hypot(old_pos[0] - self.goal[0], old_pos[1] - self.goal[1])
        new_dist = math.hypot(self.agent_pos[0] - self.goal[0], self.agent_pos[1] - self.goal[1])
        
        # 核心奖励：靠近目标给正奖励，远离给负奖励
        reward = (old_dist - new_dist) * 0.5  # 距离缩短 -> 正奖励
        if new_dist < old_dist:
            reward += 0.3  # 额外奖励
        else:
            reward -= 0.2  # 额外惩罚

        # 到达目标
        if self.agent_pos == self.goal:
            reward = 20.0
            self.done = True
            return self._get_state(), reward, True, {'reason': 'goal'}

        # 超时
        if self.steps >= 150:
            self.done = True
            return self._get_state(), reward, True, {'reason': 'max_steps'}

        return self._get_state(), reward, False, {}

    def render(self, path=None, ax=None, title=""):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        ax.clear()
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        for i in range(self.size+1):
            ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
        for (ox, oy) in self.obstacles:
            rect = Rectangle((ox-0.5, oy-0.5), 1, 1, facecolor='black', edgecolor='none')
            ax.add_patch(rect)
        if path:
            path_arr = np.array(path)
            if len(path_arr) > 1:
                ax.plot(path_arr[:,1], path_arr[:,0], 'r-', linewidth=2.5, alpha=0.8)
            ax.plot(path_arr[-1,1], path_arr[-1,0], 'r^', markersize=10)
        ax.plot(self.start[1], self.start[0], 'bo', markersize=12, label='Start')
        ax.plot(self.goal[1], self.goal[0], 'g*', markersize=14, label='Goal')
        ax.set_title(title)
        ax.legend()
        plt.draw()
        plt.pause(0.01)