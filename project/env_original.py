import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 障碍物数据（与 env1 完全一致）
OBSTACLES = set([
    (1,13), (0,10), (1,10), (2,10),
    (1,7), (2,7), (3,7), (4,7),
    (3,0), (3,1),
    (3,2), (3,3),(4,2), (5,2),(6,2),
    (9,2), (9,3), (9,4), (7,7), (8,7),
    (9,7), (10,7), (7,2), (7,3), (7,4),
    (12,5), (13,5), (14,5),
    (12,8), (13,8), (14,8),
    (11,11), (11,12), (11,13),
    (11,14),
])

class GridEnv:
    def __init__(self, seed=42):
        self.size = 15
        self.obstacles = OBSTACLES
        self.start = (1, 1)
        self.goal = (13, 13)
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

    # =========================================================
    # step 函数：原始瞬移版（无平滑插值）
    # =========================================================
    def step(self, action):
        self.steps += 1
        dx, dy = self.actions[action]
        old_pos = self.agent_pos
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy

        # 边界检查
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            self.done = True
            return self._get_state(), -10.0, True, {'reason': 'boundary'}

        # 障碍物检查
        if (new_x, new_y) in self.obstacles:
            self.done = True
            return self._get_state(), -10.0, True, {'reason': 'collision'}

        # 瞬移跳跃
        self.agent_pos = (new_x, new_y)

        # 奖励计算（与 env1 完全一致）
        old_dist = math.hypot(old_pos[0] - self.goal[0], old_pos[1] - self.goal[1])
        new_dist = math.hypot(self.agent_pos[0] - self.goal[0], self.agent_pos[1] - self.goal[1])
        reward = (old_dist - new_dist) * 2.0

        if self.agent_pos == self.goal:
            self.done = True
            return self._get_state(), 50.0, True, {'reason': 'goal'}

        if self.steps >= 300:
            self.done = True
            return self._get_state(), reward, True, {'reason': 'max_steps'}

        return self._get_state(), reward, False, {}

    # =========================================================
    # 渲染函数（沿用你的风格）
    # =========================================================
    def _get_cell_path(self, path):
        if not path:
            return []
        cell_path = []
        prev_cell = None
        for pos in path:
            cx, cy = int(pos[0]), int(pos[1])
            if prev_cell is None:
                cell_path.append((cx, cy))
            else:
                x0, y0 = prev_cell
                x1, y1 = cx, cy
                steps = max(abs(x1 - x0), abs(y1 - y0), 1)
                for i in range(1, steps + 1):
                    icx = x0 + round((x1 - x0) * i / steps)
                    icy = y0 + round((y1 - y0) * i / steps)
                    cell_path.append((icx, icy))
            prev_cell = (cx, cy)
        return cell_path

    def render(self, path=None, ax=None, title=""):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
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
            cell_path = self._get_cell_path(path)
            if cell_path:
                for (px, py) in cell_path:
                    if (px, py) != (int(self.start[0]), int(self.start[1])) and \
                       (px, py) != (int(self.goal[0]), int(self.goal[1])):
                        rect = Rectangle((px - 0.35, py - 0.35), 0.7, 0.7,
                                       facecolor='red', alpha=0.3, edgecolor='none', zorder=1)
                        ax.add_patch(rect)
                path_arr = np.array(cell_path, dtype=float)
                ax.plot(path_arr[:, 0], path_arr[:, 1],
                       color='red', linewidth=2.5, alpha=0.8, zorder=3,
                       solid_capstyle='round')
                ax.plot(path_arr[-1, 0], path_arr[-1, 1], 'r^', markersize=10,
                       markerfacecolor='red', markeredgecolor='white', markeredgewidth=1.5, zorder=5)

        ax.plot(self.start[0], self.start[1], 'o',
               markersize=12, markerfacecolor='blue', markeredgecolor='blue', label='Start', zorder=5)
        ax.plot(self.goal[0], self.goal[1], 's',
               markersize=12, markerfacecolor='green', markeredgecolor='green', label='Goal', zorder=5)

        ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.01)

    def show_map(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.render(ax=ax, title="原始环境 (瞬移版)")
        plt.show()


if __name__ == "__main__":
    env = GridEnv()
    env.show_map()
    print(f"障碍物数量: {len(env.obstacles)}, 起点={env.start}, 终点={env.goal}")