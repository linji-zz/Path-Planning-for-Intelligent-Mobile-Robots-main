import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# Hyperparameters
# ============================================================
MAP_SIZE = 30
ACTION_SPACE = 8  # 8-directional movement

GAMMA = 0.9
LR = 0.005
BATCH_SIZE = 128
BUFFER_SIZE = 2000
EPSILON_START = 0.9
EPSILON_END = 0.009
EPSILON_DECAY_STEPS = 10000
TARGET_UPDATE = 100
TOTAL_TRAINING_STEPS = 10000
MAX_STEPS_PER_EPISODE = 200
MIN_REPLAY_SIZE = 1000

# Reward parameters
LAMBDA_1 = 10    # Distance change (approaching)
LAMBDA_2 = -5    # Distance change (moving away)
LAMBDA_3 = -2    # Obstacle in outer layer
LAMBDA_4 = 4     # Target in inner layer
LAMBDA_5 = -10   # Step penalty
LAMBDA_6 = 50    # Target reached
LAMBDA_7 = -50   # Collision / out-of-bounds

# PER parameters
PER_ALPHA = 0.6
PER_ZETA = 0.4

# DWA+PER joint weighting (now both based on true DWA simulation)
DWA_WEIGHT = 0.3
TD_WEIGHT = 0.7

# DWA parameters (for true trajectory simulation)
DWA_V_MAX = 1.0
DWA_W_MAX = 1.0
DWA_V_ACC = 2.0
DWA_W_ACC = 2.0
DWA_EVAL_TIME = 1.0
DWA_DT = 0.1
DWA_SAMPLES = 20

# Environment maps
ENV_A_OBSTACLES = [
    (3, 26), (4, 26), (5, 26), (6, 26), (7, 26), (8, 26), (9, 26),
    (3, 25), (4, 25), (5, 25), (6, 25), (7, 25),
    (3, 24), (4, 24),
    (3, 23), (4, 23),
    (3, 22), (4, 22),
    (2, 25),
    (4, 19), (5, 19), (6, 19), (7, 19), (8, 19), (9, 19),
    (5, 15), (6, 15),
    (5, 14), (6, 14),
    (5, 13), (6, 13),
    (5, 12), (6, 12),
    (1, 8), (2, 8),
    (1, 7), (2, 7), (3, 7),
    (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
    (12, 24), (13, 23), (14, 22), (15, 21),
    (5, 10), (6, 10), (7, 10), (8, 10), (9, 10),
    (10, 10), (11, 10), (12, 10), (13, 10), (14, 10),
    (15, 6), (16, 6), (17, 6), (18, 6), (19, 6),
    (15, 5), (16, 5), (17, 5), (18, 5), (19, 5),
    (16, 17), (16, 16), (16, 15), (16, 14),
    (16, 13), (16, 12), (16, 11),
    (20, 27), (20, 26), (20, 25), (20, 24),
    (20, 23), (20, 22),
    (24, 23), (25, 23), (26, 23), (27, 23), (28, 23),
    (23, 12), (24, 12), (25, 12), (26, 12),
    (27, 12), (28, 12),
]

ENV_B_OBSTACLES = [
    (8, 8), (8, 9), (8, 10), (8, 11), (8, 12),
    (9, 8), (10, 8), (11, 8), (12, 8),
    (12, 9), (12, 10), (12, 11), (12, 12),
    (18, 18), (18, 19), (18, 20), (18, 21),
    (19, 18), (20, 18), (21, 18),
    (21, 19), (21, 20), (21, 21),
    (3, 15), (3, 16), (3, 17),
    (25, 5), (25, 6), (25, 7), (25, 8),
    (10, 20), (11, 20), (12, 20),
    (5, 3), (6, 3), (7, 3),
    (22, 10), (23, 10), (24, 10),
    (15, 14), (15, 15), (15, 16),
    (2, 22), (3, 22), (4, 22),
    (27, 15), (27, 16), (27, 17),
    (10, 5), (11, 5), (12, 5),
    (20, 3), (21, 3), (22, 3),
    (6, 25), (7, 25), (8, 25),
    (28, 25), (28, 26), (28, 27),
    (1, 10), (1, 11), (1, 12),
    (14, 25), (15, 25), (16, 25),
    (9, 15), (10, 15),
    (18, 5), (19, 5), (20, 5),
    (23, 22), (24, 22),
    (13, 12), (14, 12),
]

DYNAMIC_OBSTACLES_DEF = [
    {'pos': (10, 10), 'direction': (-1, 1)},
    {'pos': (20, 15), 'direction': (0, 1)},
]


# ============================================================
# Grid Map Environment (unchanged)
# ============================================================
class GridMapEnv:
    """30x30 grid map for robot path planning."""

    ACTIONS_8 = [
        (0, 1),   # 0: Up
        (1, 0),   # 1: Right
        (0, -1),  # 2: Down
        (-1, 0),  # 3: Left
        (1, 1),   # 4: UpperRight
        (1, -1),  # 5: LowerRight
        (-1, -1), # 6: LowerLeft
        (-1, 1),  # 7: UpperLeft
    ]

    def __init__(self, obstacle_list, start=(6, 4), goal=(24, 26),
                 dynamic_obstacles=None, random_start_goal=False):
        self.obstacle_list = set(obstacle_list)
        self.default_start = start
        self.default_goal = goal
        self.dynamic_obstacles_def = dynamic_obstacles
        self.random_start_goal = random_start_goal
        self.actions = self.ACTIONS_8
        self.action_dim = 8
        self.reset()

    def reset(self, start=None, goal=None):
        if start is None:
            start = self.default_start
        if goal is None:
            goal = self.default_goal

        if self.random_start_goal:
            start, goal = self._random_start_goal()

        self.agent_pos = np.array(start, dtype=float)
        self.goal_pos = np.array(goal, dtype=float)
        self.start_pos = np.array(start, dtype=float)
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self.last_path_cells = []

        self.dynamic_obstacles = []
        if self.dynamic_obstacles_def:
            for obs in self.dynamic_obstacles_def:
                self.dynamic_obstacles.append({
                    'pos': np.array(obs['pos'], dtype=float),
                    'direction': np.array(obs['direction'], dtype=float),
                })

        return self._get_state()

    def _random_start_goal(self):
        while True:
            start = (random.randint(1, MAP_SIZE - 2), random.randint(1, MAP_SIZE - 2))
            goal = (random.randint(1, MAP_SIZE - 2), random.randint(1, MAP_SIZE - 2))
            if (start not in self.obstacle_list and
                goal not in self.obstacle_list and
                start != goal):
                return start, goal

    def _get_all_obstacles(self):
        all_obs = set(self.obstacle_list)
        for dyn in self.dynamic_obstacles:
            all_obs.add((int(dyn['pos'][0]), int(dyn['pos'][1])))
        return all_obs

    def step(self, action):
        self.steps += 1
        old_pos = self.agent_pos.copy()

        dx, dy = self.actions[action]
        new_pos = self.agent_pos + np.array([dx, dy])

        all_obstacles = self._get_all_obstacles()
        x0, y0 = int(self.agent_pos[0]), int(self.agent_pos[1])
        x1, y1 = int(new_pos[0]), int(new_pos[1])

        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(1, steps + 1):
            cx = x0 + round((x1 - x0) * i / steps)
            cy = y0 + round((y1 - y0) * i / steps)

            out_of_bounds = (cx < 0 or cx >= MAP_SIZE or cy < 0 or cy >= MAP_SIZE)
            collision = (cx, cy) in all_obstacles

            if out_of_bounds or collision:
                self.agent_pos = old_pos
                self.done = True
                reward = LAMBDA_7
                return self._get_state(), reward, True, {'reason': 'collision'}

        self.agent_pos = new_pos

        self.last_path_cells = []
        for i in range(1, steps + 1):
            cx = x0 + round((x1 - x0) * i / steps)
            cy = y0 + round((y1 - y0) * i / steps)
            self.last_path_cells.append((cx, cy))

        reward = self._compute_reward(old_pos, all_obstacles)
        self.total_reward += reward

        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        if dist_to_goal < 1.5:
            self.done = True
            reward += LAMBDA_6
            return self._get_state(), reward, True, {'reason': 'goal'}

        if self.steps >= MAX_STEPS_PER_EPISODE:
            self.done = True
            return self._get_state(), reward, True, {'reason': 'max_steps'}

        self._update_dynamic_obstacles()

        return self._get_state(), reward, False, {}

    def _update_dynamic_obstacles(self):
        for obs in self.dynamic_obstacles:
            obs['pos'] += obs['direction']
            if obs['pos'][0] <= 0 or obs['pos'][0] >= MAP_SIZE - 1:
                obs['direction'][0] *= -1
            if obs['pos'][1] <= 0 or obs['pos'][1] >= MAP_SIZE - 1:
                obs['direction'][1] *= -1

    def _compute_reward(self, old_pos, all_obstacles):
        reward = 0

        old_dist = np.linalg.norm(old_pos - self.goal_pos)
        new_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        if new_dist < old_dist:
            reward += LAMBDA_1
        elif new_dist > old_dist:
            reward += LAMBDA_2
        else:
            reward += LAMBDA_1

        obs_local = self._get_local_observation(all_obstacles)
        if obs_local['target_in_inner']:
            reward += LAMBDA_4
        elif obs_local['inner'] > 0:
            reward += 2 * LAMBDA_3
        elif obs_local['outer'] > 0:
            reward += LAMBDA_3

        reward += LAMBDA_5

        return reward

    def _get_local_observation(self, all_obstacles):
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])

        inner_count = 0
        outer_count = 0
        target_in_inner = False

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ax + dx, ay + dy
                if (nx, ny) in all_obstacles:
                    inner_count += 1
                if (nx, ny) == (gx, gy):
                    target_in_inner = True

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) <= 1 and abs(dy) <= 1:
                    continue
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    if (nx, ny) in all_obstacles:
                        outer_count += 1

        return {
            'inner': inner_count,
            'outer': outer_count,
            'target_in_inner': target_in_inner,
        }

    def _get_state(self):
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        all_obstacles = self._get_all_obstacles()
        local_obs = self._get_local_observation(all_obstacles)

        local_grid = np.zeros(25)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = ax + dx, ay + dy
                idx = (dx + 2) * 5 + (dy + 2)
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    if (nx, ny) in all_obstacles:
                        local_grid[idx] = 1
                    if (nx, ny) == (int(self.goal_pos[0]), int(self.goal_pos[1])):
                        local_grid[idx] = 0.5

        state = np.array([
            self.agent_pos[0] / MAP_SIZE,
            self.agent_pos[1] / MAP_SIZE,
            self.goal_pos[0] / MAP_SIZE,
            self.goal_pos[1] / MAP_SIZE,
            local_obs['inner'] / 8.0,
            local_obs['outer'] / 16.0,
            float(local_obs['target_in_inner']),
            *local_grid
        ], dtype=np.float32)

        return state

    def render(self, path=None, ax=None, title=""):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        ax.clear()
        ax.set_xlim(-0.5, MAP_SIZE - 0.5)
        ax.set_ylim(-0.5, MAP_SIZE - 0.5)
        ax.set_aspect('equal')

        for i in range(MAP_SIZE + 1):
            ax.axhline(y=i - 0.5, color='gray', linewidth=0.3, alpha=0.4)
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.3, alpha=0.4)

        tick_positions = [0, 5, 10, 15, 20, 25, 30]
        ax.set_xticks([t - 0.5 for t in tick_positions])
        ax.set_xticklabels(tick_positions, fontsize=8)
        ax.set_yticks([t - 0.5 for t in tick_positions])
        ax.set_yticklabels(tick_positions, fontsize=8)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)

        for (ox, oy) in self.obstacle_list:
            rect = Rectangle((ox - 0.5, oy - 0.5), 1, 1,
                           facecolor='black', edgecolor='none')
            ax.add_patch(rect)

        for obs in self.dynamic_obstacles:
            ox, oy = int(obs['pos'][0]), int(obs['pos'][1])
            rect = Rectangle((ox - 0.5, oy - 0.5), 1, 1,
                           facecolor='black', edgecolor='none')
            ax.add_patch(rect)

        if path:
            path = np.array(path)
            for i in range(len(path) - 1):
                ax.plot(path[i:i+2, 0], path[i:i+2, 1],
                       color='red', linewidth=2, alpha=0.8)
            ax.plot(path[-1, 0], path[-1, 1], 'r^', markersize=8,
                   markerfacecolor='red', markeredgecolor='white', markeredgewidth=1)

        ax.plot(self.start_pos[0], self.start_pos[1], 'o',
               markersize=10, markerfacecolor='blue', markeredgecolor='blue',
               label='Start', zorder=5)
        ax.plot(self.goal_pos[0], self.goal_pos[1], 's',
               markersize=10, markerfacecolor='green', markeredgecolor='green',
               label='Target', zorder=5)

        from matplotlib.patches import Patch
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, label='Start'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                       markersize=8, label='Target'),
            Patch(facecolor='black', edgecolor='none', label='Obstacle'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
                 framealpha=1.0, edgecolor='black', fancybox=False)

        ax.set_title(title, fontsize=11, pad=8)

        plt.draw()
        plt.pause(0.01)

    def get_cell_path(self, recorded_positions):
        if not recorded_positions:
            return []

        cell_path = []
        prev_cell = None
        for pos in recorded_positions:
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


# ============================================================
# 1. Dueling DQN Network (unchanged)
# ============================================================
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        features = self.shared(x)
        value = self.value(features)
        advantage = self.advantage(features)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ============================================================
# 2. True Dynamic Window Approach (DWA) Planner
# ============================================================
class TrueDWAPlanner:
    """
    Complete DWA planner for a differential-drive robot.
    It samples linear and angular velocities, simulates trajectories,
    and evaluates them with heading, clearance, and speed scores.
    Here we use it to score each discrete action by simulating a short
    straight-line trajectory (since our actions are discrete).
    """

    def __init__(self, v_max=DWA_V_MAX, v_min=0.0, w_max=DWA_W_MAX,
                 v_acc=DWA_V_ACC, w_acc=DWA_W_ACC, dt=DWA_DT,
                 eval_time=DWA_EVAL_TIME, n_samples=DWA_SAMPLES):
        self.v_max = v_max
        self.v_min = v_min
        self.w_max = w_max
        self.v_acc = v_acc
        self.w_acc = w_acc
        self.dt = dt
        self.eval_time = eval_time
        self.n_samples = n_samples

    def score_action(self, action_idx, agent_pos, goal, obstacles, env_actions):
        """
        Score a discrete action by simulating a trajectory along that direction
        and evaluating it with DWA criteria (heading, clearance, speed).
        Returns a score in [0, 1].
        """
        dx, dy = env_actions[action_idx]
        # 模拟沿该方向直线运动
        traj = []
        x, y = agent_pos[0], agent_pos[1]
        # 模拟步数
        steps = int(self.eval_time / self.dt)
        for _ in range(steps):
            x += dx * self.dt
            y += dy * self.dt
            # 边界和碰撞检查
            if x < 0 or x >= MAP_SIZE or y < 0 or y >= MAP_SIZE:
                break
            if (int(round(x)), int(round(y))) in obstacles:
                break
            traj.append((x, y))
        if not traj:
            return 0.0  # 立即碰撞或无效

        # 评价轨迹
        return self._evaluate_trajectory(traj, agent_pos, goal, obstacles)

    def _evaluate_trajectory(self, traj, start_pos, goal, obstacles):
        # 1. Heading: 末端朝向目标的程度
        final_pos = traj[-1]
        dx_goal = goal[0] - final_pos[0]
        dy_goal = goal[1] - final_pos[1]
        dist_goal = np.hypot(dx_goal, dy_goal)
        if dist_goal < 1e-6:
            heading = 1.0
        else:
            if len(traj) >= 2:
                dx_traj = traj[-1][0] - traj[-2][0]
                dy_traj = traj[-1][1] - traj[-2][1]
                if np.hypot(dx_traj, dy_traj) < 1e-6:
                    heading = 0.0
                else:
                    cos_angle = (dx_traj * dx_goal + dy_traj * dy_goal) / (
                        np.hypot(dx_traj, dy_traj) * dist_goal)
                    heading = max(0.0, cos_angle)
            else:
                heading = 0.0

        # 2. Clearance: 轨迹上离障碍物的最小距离（归一化）
        clearance = float('inf')
        for (x, y) in traj:
            for (ox, oy) in obstacles:
                d = np.hypot(x - ox, y - oy)
                if d < clearance:
                    clearance = d
        max_clearance = 5.0
        clearance = min(clearance / max_clearance, 1.0)

        # 3. Speed: 假定速度恒定为1.0（因为离散动作步长为1），直接给1.0
        speed = 1.0

        # 加权综合（系数可调）
        alpha = 0.5
        beta = 0.3
        gamma = 0.2
        return alpha * heading + beta * clearance + gamma * speed


# ============================================================
# 3. DWA+PER Joint Priority Replay Buffer (modified)
# ============================================================
class DWAPERBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, dwa_score=None):
        if dwa_score is not None:
            priority = max(self.max_priority,
                         dwa_score * DWA_WEIGHT + self.max_priority * TD_WEIGHT)
        else:
            priority = self.max_priority

        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size, zeta=PER_ZETA):
        if len(self.buffer) == 0:
            return None

        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-zeta)
        weights /= weights.max()

        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.LongTensor(actions).unsqueeze(1),
            'rewards': torch.FloatTensor(rewards).unsqueeze(1),
            'next_states': torch.FloatTensor(np.array(next_states)),
            'dones': torch.FloatTensor(dones).unsqueeze(1),
            'indices': indices,
            'weights': torch.FloatTensor(weights).unsqueeze(1),
        }

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = abs(td) * TD_WEIGHT + self.max_priority * DWA_WEIGHT
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 4. Improved DDQN Agent (with true DWA scoring)
# ============================================================
class ImprovedDDQNAgent:
    def __init__(self, state_dim, action_dim=8):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = DWAPERBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.dwa_planner = TrueDWAPlanner()  # 使用真正的DWA规划器

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def compute_target(self, rewards, next_states, dones):
        with torch.no_grad():
            best_actions = self.policy_net(next_states).max(dim=1, keepdim=True)[1]
            max_next_q = self.target_net(next_states).gather(1, best_actions)
            targets = rewards + GAMMA * max_next_q * (1 - dones)
        return targets

    def train_step(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None

        batch = self.replay_buffer.sample(BATCH_SIZE)
        if batch is None:
            return None

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        indices = batch['indices']
        weights = batch['weights']

        q_values = self.policy_net(states).gather(1, actions)
        targets = self.compute_target(rewards, next_states, dones)

        td_errors = (q_values - targets).detach().numpy()
        loss = (weights * (q_values - targets) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.flatten())

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        progress = self.steps_done / TOTAL_TRAINING_STEPS
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * (1 - progress)
        self.steps_done += 1

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']


# ============================================================
# Training Loop (modified to use true DWA scoring)
# ============================================================
def train(env_name='A', render=False, save_dir='./'):
    if env_name == 'A':
        obstacles = ENV_A_OBSTACLES
        start, goal = (6, 4), (24, 26)
    elif env_name == 'B':
        obstacles = ENV_B_OBSTACLES
        start, goal = (3, 4), (22, 27)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    env = GridMapEnv(obstacles, start=start, goal=goal)
    test_state = env.reset()
    state_dim = len(test_state)

    print(f"{'='*60}")
    print(f"Improved DQN: Dueling + 8-Dir + True DWA Scoring + PER")
    print(f"{'='*60}")
    print(f"Environment: {env_name} (30x30 Grid Map)")
    print(f"Start: {start}, Goal: {goal}")
    print(f"State dim: {state_dim}, Actions: 8")
    print(f"Network: Dueling DQN (Value + Advantage)")
    print(f"Replay: DWA+PER Joint Priority (using true DWA simulation)")
    print(f"Training steps: {TOTAL_TRAINING_STEPS}")
    print(f"{'='*60}")

    agent = ImprovedDDQNAgent(state_dim, action_dim=8)

    episode_rewards = []
    success_count = 0
    best_avg_reward = -float('inf')
    loss_history = []
    state = env.reset()
    path = [env.agent_pos.copy()]

    for step in range(TOTAL_TRAINING_STEPS):
        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        # --- Use true DWA to score the chosen action ---
        all_obs = env._get_all_obstacles()
        dwa_score = agent.dwa_planner.score_action(
            action, env.agent_pos, env.goal_pos,
            all_obs, env.actions
        )
        # ------------------------------------------------

        agent.replay_buffer.push(state, action, reward, next_state, done,
                                 dwa_score=dwa_score)

        path.append(env.agent_pos.copy())

        loss = agent.train_step()
        if loss is not None:
            loss_history.append(loss)

        if step % TARGET_UPDATE == 0:
            agent.update_target_network()

        agent.decay_epsilon()

        state = next_state

        if done:
            episode_rewards.append(env.total_reward)
            if info.get('reason') == 'goal':
                success_count += 1

            if len(episode_rewards) % 10 == 0:
                avg_r = np.mean(episode_rewards[-10:])
                win_rate = success_count / len(episode_rewards) * 100
                print(f"Step {step:5d} | Episodes: {len(episode_rewards):4d} | "
                      f"Avg Reward: {avg_r:6.1f} | Win Rate: {win_rate:5.1f}% | "
                      f"Epsilon: {agent.epsilon:.3f} | Loss: {loss_history[-1] if loss_history else 0:.3f}")

                if avg_r > best_avg_reward:
                    best_avg_reward = avg_r
                    agent.save(os.path.join(save_dir, f'dqn_policy_best_{env_name}.pth'))

            env.reset()
            state = env.reset()
            path = [env.agent_pos.copy()]

    agent.save(os.path.join(save_dir, f'dqn_policy_final_{env_name}.pth'))

    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Success rate: {success_count}/{len(episode_rewards)} "
          f"({success_count/max(1,len(episode_rewards))*100:.1f}%)")
    print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.1f}")
    print(f"{'='*60}")

    plot_reward_curve(episode_rewards, f'training_reward_{env_name}.png')
    if loss_history:
        plot_loss_curve(loss_history, f'training_loss_{env_name}.png')

    plot_final_result(env, agent, num_episodes=3, save_prefix=f'result_{env_name}')

    return agent, episode_rewards, loss_history


# ============================================================
# Test / Visualization (unchanged)
# ============================================================
def test_policy(policy_path, env_name='A', render=True, num_episodes=5):
    if env_name == 'A':
        obstacles = ENV_A_OBSTACLES
        start, goal = (6, 4), (24, 26)
    elif env_name == 'B':
        obstacles = ENV_B_OBSTACLES
        start, goal = (3, 4), (22, 27)
    else:
        obstacles = ENV_A_OBSTACLES
        start, goal = (6, 4), (24, 26)

    env = GridMapEnv(obstacles, start=start, goal=goal)
    test_state = env.reset()
    state_dim = len(test_state)

    agent = ImprovedDDQNAgent(state_dim, action_dim=8)
    agent.load(policy_path)

    print(f"Testing policy: {policy_path}")
    print(f"Environment: {env_name} | Actions: 8")
    print("-" * 40)

    success_count = 0
    fig, ax = plt.subplots(1, 1, figsize=(8, 8)) if render else (None, None)

    for ep in range(num_episodes):
        state = env.reset()
        path = [env.agent_pos.copy()]
        done = False

        while not done:
            with torch.no_grad():
                q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = q_values.argmax().item()

            state, reward, done, info = env.step(action)
            path.append(env.agent_pos.copy())

            if render:
                env.render(
                    path=path, ax=ax,
                    title=f"Episode {ep+1} | Reward: {env.total_reward:.0f} | "
                          f"{'SUCCESS' if info.get('reason') == 'goal' else 'Failed'}"
                )

        if info.get('reason') == 'goal':
            success_count += 1
            print(f"Episode {ep+1}: SUCCESS | Reward: {env.total_reward:.0f} | Steps: {env.steps}")
        else:
            print(f"Episode {ep+1}: FAILED ({info.get('reason')}) | Reward: {env.total_reward:.0f}")

    print(f"{'='*40}")
    print(f"Success rate: {success_count}/{num_episodes}")

    if render:
        plt.show()


def test_dynamic_env(policy_path, num_episodes=5):
    env = GridMapEnv(
        obstacle_list=ENV_A_OBSTACLES,
        start=(6, 4), goal=(24, 26),
        dynamic_obstacles=DYNAMIC_OBSTACLES_DEF,
    )

    test_state = env.reset()
    state_dim = len(test_state)

    agent = ImprovedDDQNAgent(state_dim, action_dim=8)
    agent.load(policy_path)

    print(f"Testing in DYNAMIC environment | Actions: 8")
    print(f"Policy: {policy_path}")
    print("-" * 40)

    success_count = 0
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for ep in range(num_episodes):
        state = env.reset()
        path = [env.agent_pos.copy()]
        done = False

        while not done:
            with torch.no_grad():
                q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = q_values.argmax().item()

            state, reward, done, info = env.step(action)
            path.append(env.agent_pos.copy())

            env.render(
                path=path, ax=ax,
                title=f"Dynamic | Ep {ep+1} | Reward: {env.total_reward:.0f} | "
                      f"{'SUCCESS' if info.get('reason') == 'goal' else 'Failed'}"
            )

        if info.get('reason') == 'goal':
            success_count += 1
            print(f"Episode {ep+1}: SUCCESS | Reward: {env.total_reward:.0f} | Steps: {env.steps}")
        else:
            print(f"Episode {ep+1}: FAILED ({info.get('reason')}) | Reward: {env.total_reward:.0f}")

    print(f"{'='*40}")
    print(f"Dynamic env success rate: {success_count}/{num_episodes}")
    plt.show()


# ============================================================
# Plot functions (unchanged)
# ============================================================
def plot_reward_curve(rewards, save_path='training_reward.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(rewards, alpha=0.5, color='blue', label='Episode Reward')
    window = 10
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), avg, color='red', linewidth=2, label=f'{window}-Episode Avg')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Training Rewards', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Reward curve saved to: {save_path}")
    plt.show()

def plot_loss_curve(losses, save_path='training_loss.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(losses, alpha=0.3, color='blue', label='Loss')
    window = 50
    if len(losses) >= window:
        avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses)), avg_loss, color='red', linewidth=2, label=f'{window}-Step Avg')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to: {save_path}")
    plt.show()

def plot_grid_map(env, path=None, title="", save_path='grid_map.png', show_label='(a)'):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlim(-0.5, MAP_SIZE - 0.5)
    ax.set_ylim(-0.5, MAP_SIZE - 0.5)
    ax.set_aspect('equal')
    for i in range(MAP_SIZE + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.3, alpha=0.4)
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.3, alpha=0.4)
    tick_positions = [0, 5, 10, 15, 20, 25, 30]
    ax.set_xticks([t - 0.5 for t in tick_positions])
    ax.set_xticklabels(tick_positions, fontsize=9)
    ax.set_yticks([t - 0.5 for t in tick_positions])
    ax.set_yticklabels(tick_positions, fontsize=9)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    for (ox, oy) in env.obstacle_list:
        rect = Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor='black', edgecolor='none')
        ax.add_patch(rect)
    for obs in env.dynamic_obstacles:
        ox, oy = int(obs['pos'][0]), int(obs['pos'][1])
        rect = Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor='black', edgecolor='none')
        ax.add_patch(rect)
    if path:
        if len(path) > 0 and not isinstance(path[0], tuple):
            cell_path = env.get_cell_path(path)
        else:
            cell_path = path
        if cell_path:
            for (px, py) in cell_path:
                if (px, py) != (int(env.start_pos[0]), int(env.start_pos[1])) and \
                   (px, py) != (int(env.goal_pos[0]), int(env.goal_pos[1])):
                    rect = Rectangle((px - 0.35, py - 0.35), 0.7, 0.7,
                                   facecolor='red', alpha=0.3, edgecolor='none', zorder=1)
                    ax.add_patch(rect)
            path_arr = np.array(cell_path, dtype=float)
            ax.plot(path_arr[:, 0], path_arr[:, 1], color='red', linewidth=2.5, alpha=0.8, zorder=3, solid_capstyle='round')
            ax.plot(path_arr[-1, 0], path_arr[-1, 1], 'r^', markersize=10, markerfacecolor='red', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.plot(env.start_pos[0], env.start_pos[1], 'o', markersize=12, markerfacecolor='blue', markeredgecolor='blue', label='Start', zorder=5)
    ax.plot(env.goal_pos[0], env.goal_pos[1], 's', markersize=12, markerfacecolor='green', markeredgecolor='green', label='Target', zorder=5)
    if show_label:
        ax.set_xlabel(f'{show_label}', fontsize=14, fontweight='bold', labelpad=10)
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Target'),
        Patch(facecolor='black', edgecolor='none', label='Obstacle'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=1.0, edgecolor='black', fancybox=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grid map saved to: {save_path}")
    plt.show()

def plot_final_result(env, agent, num_episodes=3, save_prefix='result'):
    print(f"\n{'='*50}")
    print(f"Visualizing Final Policy")
    print(f"{'='*50}")
    for ep in range(num_episodes):
        state = env.reset()
        path = [env.agent_pos.copy()]
        done = False
        while not done:
            with torch.no_grad():
                q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = q_values.argmax().item()
            state, reward, done, info = env.step(action)
            path.append(env.agent_pos.copy())
        status = 'SUCCESS' if info.get('reason') == 'goal' else f"FAILED ({info.get('reason')})"
        print(f"Episode {ep+1}: {status} | Reward: {env.total_reward:.0f} | Steps: {env.steps}")
        save_path = f'{save_prefix}_ep{ep+1}.png'
        plot_grid_map(env, path=path,
                     title=f"Episode {ep+1}: {status} | Reward: {env.total_reward:.0f} | Steps: {env.steps}",
                     save_path=save_path, show_label=None)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN with True DWA Scoring")
    parser.add_argument("--mode", choices=["train", "test", "dynamic", "plot"], default="train")
    parser.add_argument("--env", choices=["A", "B"], default="A")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="./")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering during test")
    args = parser.parse_args()

    if args.mode == "train":
        agent, rewards, losses = train(env_name=args.env, save_dir=args.save_dir)
        if rewards:
            plot_reward_curve(rewards, f'training_reward_{args.env}.png')
        if losses:
            plot_loss_curve(losses, f'training_loss_{args.env}.png')

    elif args.mode == "test":
        policy = args.policy or f'dqn_policy_best_{args.env}.pth'
        if not os.path.exists(policy):
            policy = f'dqn_policy_final_{args.env}.pth'
        test_policy(policy, env_name=args.env,
                    render=not args.no_render, num_episodes=args.episodes)

    elif args.mode == "dynamic":
        policy = args.policy or f'dqn_policy_best_{args.env}.pth'
        if not os.path.exists(policy):
            policy = f'dqn_policy_final_{args.env}.pth'
        test_dynamic_env(policy, num_episodes=args.episodes)

    elif args.mode == "plot":
        env = GridMapEnv(
            ENV_A_OBSTACLES if args.env == 'A' else ENV_B_OBSTACLES
        )
        env.reset()
        label = '(a)' if args.env == 'A' else '(b)'
        plot_grid_map(env, title=f"Environment {args.env}",
                     save_path=f'env_{args.env}.png', show_label=label)