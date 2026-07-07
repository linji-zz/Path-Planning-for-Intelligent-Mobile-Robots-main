import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =====================================================
# 环境2: 20x20, 障碍物密集分布
# =====================================================

OBSTACLES = set([
    (0, 5), (1, 15), (2, 0), (3, 0), (2, 1), (2, 4),
    (1, 18), (2, 14), (2, 15),(2, 16), (2, 8), (2, 9), (2, 10), (3, 8), (4, 8),
    (4, 9), (4, 10), (5, 3), (7, 3), (6, 2), (6, 3), (6, 4),
    (6, 6),(6, 7),
    (6, 12), (6, 13), (7, 12), (8, 11), (6, 17), (9, 16),
    (10, 12), (9,9), (10, 9), (10, 8), (11, 8),
    (12, 14), (12, 18), (13, 18),(14, 18), (14, 17),
    (16, 14), (16, 15), (16, 16), (17, 15), (18, 12),(18, 11),(18, 12),
    (13, 11), (13, 10),(14, 10), (16, 11), (19, 3), (19, 4), (18, 3),(18, 4), 
    (12, 4), (13, 6),(14, 6),(12, 0), (12, 1),  
])

class GridEnv:
    def __init__(self, seed=42):
        self.size = 20
        self.obstacles = OBSTACLES
        self.start = (1, 1)
        self.goal = (18, 18)
        self.actions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.action_dim = 8
        random.seed(seed)
        np.random.seed(seed)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        dx = (self.goal[0] - self.agent_pos[0]) / self.size
        dy = (self.goal[1] - self.agent_pos[1]) / self.size
        dist = math.hypot(dx * self.size, dy * self.size) / self.size
        obs_flags = []
        x, y = self.agent_pos
        for dx_a, dy_a in self.actions:
            nx, ny = x + dx_a, y + dy_a
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                obs_flags.append(1.0)
            elif (nx, ny) in self.obstacles:
                obs_flags.append(1.0)
            else:
                obs_flags.append(0.0)
        return np.array([dx, dy, dist] + obs_flags, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx, dy = self.actions[action]
        old_pos = self.agent_pos
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            reward = -10.0
            self.done = True
            return self._get_state(), reward, True, {'reason': 'boundary'}
        new_pos = (new_x, new_y)
        if new_pos in self.obstacles:
            reward = -10.0
            self.done = True
            return self._get_state(), reward, True, {'reason': 'collision'}
        self.agent_pos = new_pos
        old_dist = math.hypot(old_pos[0] - self.goal[0], old_pos[1] - self.goal[1])
        new_dist = math.hypot(self.agent_pos[0] - self.goal[0], self.agent_pos[1] - self.goal[1])
        reward = (old_dist - new_dist) * 2.0
        if self.agent_pos == self.goal:
            reward = 50.0
            self.done = True
            return self._get_state(), reward, True, {'reason': 'goal'}
        if self.steps >= 300:
            self.done = True
            return self._get_state(), reward, True, {'reason': 'max_steps'}
        return self._get_state(), reward, False, {}

    def render(self, path=None, ax=None, title=""):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        ax.clear()
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        for i in range(self.size + 1):
            ax.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for (ox, oy) in self.obstacles:
            rect = Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor='black', edgecolor='none')
            ax.add_patch(rect)
        if path:
            path_arr = np.array(path)
            if len(path_arr) > 1:
                ax.plot(path_arr[:, 1], path_arr[:, 0], 'r-', linewidth=2.5, alpha=0.8)
            ax.plot(path_arr[-1, 1], path_arr[-1, 0], 'r^', markersize=10)
        ax.plot(self.start[1], self.start[0], 'bo', markersize=12, label='Start')
        ax.plot(self.goal[1], self.goal[0], 'go', markersize=12, label='Goal')
        ax.set_title(title)
        ax.legend()
        plt.draw()
        plt.pause(0.01)

    def show_map(self):
        fig, ax = plt.subplots(figsize=(6,6))
        self.render(ax=ax, title=f"环境2 (20×20)")
        plt.show()


if __name__ == "__main__":
    env = GridEnv()
    env.show_map()
    print(f"环境2: 20x20, 障碍物数量: {len(env.obstacles)}, 起点={env.start}, 终点={env.goal}")