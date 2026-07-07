"""
严格复现论文《基于改进DQN的移动机器人避障路径规划》
使用从论文图6中人工重构的三种固定地图
田箫源, 董秀成. 中国惯性技术学报, 2024.
"""

import numpy as np
import random
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

# ============================================================
# 超参数（严格遵循论文表2）
# ============================================================
GRID_SIZE = 25
ACTION_DIM = 8
EPISODES = 1000
MAX_STEPS = 200
LEARNING_RATE = 0.01
GAMMA = 0.9
BATCH_SIZE = 320
MEMORY_CAPACITY = 3000
TARGET_UPDATE_FREQ = 100
VISIT_LIMIT = 380

# 奖励参数（论文未明确数值，根据表4反推）
C1 = 10.0    # 到达目标
C2 = -10.0   # 碰撞
C4 = 1.0     # 探索奖励系数
C5 = 1.0     # 启发奖励系数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 论文图6中的三种环境（根据图13-15路径图人工重构）
# ============================================================

# 环境1（图6a / 图13）- 最简单，路径步数27
ENV1_OBSTACLES = set([
    # 左上障碍物块 (3,3)-(7,7) 的L形
    (3,3), (4,3), (5,3), (6,3), (7,3),
    (3,4), (4,4), (5,4), (6,4), (7,4),
    (3,5), (4,5), (5,5), (6,5), (7,5),
    (3,6), (4,6), (5,6),
    (3,7), (4,7),
    
    # 中部障碍物块 (10,10)-(15,15) 的L形
    (10,10), (11,10), (12,10), (13,10), (14,10), (15,10),
    (10,11), (11,11), (12,11), (13,11), (14,11), (15,11),
    (10,12), (11,12), (12,12),
    (10,13), (11,13),
    (10,14),
    
    # 右下障碍物块 (17,17)-(22,22) 的L形
    (17,17), (18,17), (19,17), (20,17), (21,17), (22,17),
    (17,18), (18,18), (19,18), (20,18), (21,18), (22,18),
    (17,19), (18,19), (19,19),
    (17,20), (18,20),
    (17,21),
    
    # 额外分散障碍物（使地图更丰富）
    (8,8), (8,9), (9,8), (9,9),
    (15,15), (16,15), (15,16), (16,16),
    (5,20), (6,20), (7,20),
    (20,5), (20,6), (20,7),
])

# 环境2（图6b / 图14）- 中等复杂度，路径步数26
ENV2_OBSTACLES = set([
    # 左侧垂直墙 (x=5, y=3~21)
    (5, y) for y in range(3, 22)
] + [
    # 右侧垂直墙 (x=19, y=3~21)
    (19, y) for y in range(3, 22)
] + [
    # 顶部水平墙 (y=5, x=5~19)
    (x, 5) for x in range(5, 20)
] + [
    # 底部水平墙 (y=19, x=5~19)
    (x, 19) for x in range(5, 20)
] + [
    # 中间四个小障碍物群（形成通道）
    (10, 10), (11, 10), (12, 10),
    (10, 14), (11, 14), (12, 14),
    (14, 10), (15, 10),
    (14, 14), (15, 14),
])

# 环境3（图15）- 蛇形迷宫，路径步数26
ENV3_OBSTACLES = set([
    # 蛇形障碍物：形成蜿蜒通道
    
    # 第一段水平墙 (y=3, x=3~21)
    (x, 3) for x in range(3, 22)
] + [
    # 第二段垂直墙 (x=3, y=3~10)
    (3, y) for y in range(3, 11)
] + [
    # 第三段水平墙 (y=9, x=3~12)
    (x, 9) for x in range(3, 13)
] + [
    # 第四段垂直墙 (x=11, y=9~16)
    (11, y) for y in range(9, 17)
] + [
    # 第五段水平墙 (y=15, x=5~20)
    (x, 15) for x in range(5, 21)
] + [
    # 第六段垂直墙 (x=19, y=15~22)
    (19, y) for y in range(15, 23)
] + [
    # 第七段水平墙 (y=21, x=3~22)
    (x, 21) for x in range(3, 23)
] + [
    # 左侧垂直墙 (x=5, y=3~21)
    (5, y) for y in range(3, 22)
] + [
    # 右侧垂直墙 (x=15, y=5~15)
    (15, y) for y in range(5, 16)
] + [
    # 额外障碍物（使通道变窄）
    (7, 12), (8, 12), (9, 12),
    (13, 12), (14, 12),
    (7, 18), (8, 18), (9, 18),
])

# 环境选择
ENV_MAP = {
    '1': ENV1_OBSTACLES,
    '2': ENV2_OBSTACLES,
    '3': ENV3_OBSTACLES,
}

# ============================================================
# 栅格环境（使用固定地图）
# ============================================================
class GridEnv:
    def __init__(self, env_id='1'):
        """
        env_id: '1', '2', '3' 对应论文中的三种环境
        """
        self.size = GRID_SIZE
        self.actions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.action_dim = len(self.actions)
        
        # 使用固定地图
        self.obstacles = ENV_MAP[env_id]
        self.start = (1, 1)
        self.goal = (23, 23)
        
        print(f"环境 {env_id} 初始化完成，障碍物数量: {len(self.obstacles)}")
        print(f"起点: {self.start}, 终点: {self.goal}")
        
        self.reset()
    
    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self.visit_counts = {}
        self._update_visit_count()
        return self._get_state()
    
    def _update_visit_count(self):
        pos = self.agent_pos
        self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1
    
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
        
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            reward = C2
            self.done = True
            return self._get_state(), reward, True, {'reason': 'boundary'}
        
        new_pos = (new_x, new_y)
        if new_pos in self.obstacles:
            reward = C2
            self.done = True
            return self._get_state(), reward, True, {'reason': 'collision'}
        
        self.agent_pos = new_pos
        self._update_visit_count()
        
        reward = self._compute_reward(old_pos)
        self.total_reward += reward
        
        if self.agent_pos == self.goal:
            reward += C1
            self.done = True
            return self._get_state(), reward, True, {'reason': 'goal'}
        
        if self.steps >= MAX_STEPS:
            self.done = True
            return self._get_state(), reward, True, {'reason': 'max_steps'}
        
        return self._get_state(), reward, False, {}
    
    def _compute_reward(self, old_pos):
        reward = 0.0
        
        # 探索奖励 (论文公式7)
        n_s = self.visit_counts.get(self.agent_pos, 1)
        n_s_old = self.visit_counts.get(old_pos, 1)
        if n_s <= VISIT_LIMIT and n_s_old <= VISIT_LIMIT:
            explore_reward = max(0.0, 1.0/n_s - 1.0/n_s_old)
            reward += C4 * explore_reward
        
        # 启发奖励 (论文公式9-10)
        old_dist = self._euclidean_dist(old_pos, self.goal)
        new_dist = self._euclidean_dist(self.agent_pos, self.goal)
        if new_dist < old_dist:
            reward += C5 * (old_dist - new_dist)
        
        return reward
    
    def _euclidean_dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def render(self, path=None, ax=None, title=""):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7,7))
        ax.clear()
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        ax.set_aspect('equal')
        for i in range(self.size+1):
            ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # 绘制障碍物
        for (ox, oy) in self.obstacles:
            rect = Rectangle((ox-0.5, oy-0.5), 1, 1, facecolor='black', edgecolor='none')
            ax.add_patch(rect)
        
        # 绘制路径
        if path:
            path_arr = np.array(path)
            if len(path_arr) > 1:
                ax.plot(path_arr[:,1], path_arr[:,0], color='red', linewidth=2.5, alpha=0.8)
            ax.plot(path_arr[-1,1], path_arr[-1,0], 'r^', markersize=10, markerfacecolor='red')
        
        # 起点 (蓝色圆点)
        ax.plot(self.start[1], self.start[0], 'o', markersize=12, 
                markerfacecolor='blue', markeredgecolor='blue', label='Start')
        # 终点 (绿色星号)
        ax.plot(self.goal[1], self.goal[0], '*', markersize=14, 
                markerfacecolor='green', markeredgecolor='green', label='Goal')
        
        # 设置坐标轴刻度（与论文一致）
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_xlabel('x/m', fontsize=12)
        ax.set_ylabel('y/m', fontsize=12)
        ax.legend(loc='upper right')
        ax.set_title(title, fontsize=12)
        plt.draw()
        plt.pause(0.01)
    
    def show_map(self):
        """显示地图，用于对照论文图6"""
        fig, ax = plt.subplots(figsize=(7,7))
        self.render(ax=ax, title="Map (对照论文图6)")
        plt.show()

# ============================================================
# DQN 网络（论文图8：1875→224→8）
# ============================================================
class DQN(nn.Module):
    def __init__(self, input_dim=3*GRID_SIZE*GRID_SIZE, output_dim=ACTION_DIM):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 224)
        self.fc2 = nn.Linear(224, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ============================================================
# 障碍学习表
# ============================================================
class ObstacleTable:
    def __init__(self):
        self.table = {}
    
    def add_bad_action(self, pos, action):
        if pos not in self.table:
            self.table[pos] = set()
        self.table[pos].add(action)
    
    def get_valid_actions(self, pos):
        if pos in self.table:
            return [a for a in range(ACTION_DIM) if a not in self.table[pos]]
        else:
            return list(range(ACTION_DIM))

# ============================================================
# 经验回放
# ============================================================
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.memory)

# ============================================================
# OREDQN 智能体
# ============================================================
class OREDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.obstacle_table = ObstacleTable()
        self.steps_done = 0
        self.epsilon = 1.0
    
    def select_action(self, state, env):
        total_steps_max = EPISODES * MAX_STEPS
        progress = min(self.steps_done / total_steps_max, 1.0)
        if progress > 0:
            utilization = 0.5 * math.log(6.389 * progress + 1)
        else:
            utilization = 0.0
        utilization = min(utilization, 1.0)
        epsilon = 1.0 - utilization
        epsilon = max(0.01, epsilon)
        self.epsilon = epsilon
        
        pos = env.agent_pos
        valid_actions = self.obstacle_table.get_valid_actions(pos)
        if not valid_actions:
            valid_actions = list(range(self.action_dim))
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_t)
            q_values = q_values.cpu().numpy().flatten()
            valid_q = [q_values[a] for a in valid_actions]
            best_idx = np.argmax(valid_q)
            return valid_actions[best_idx]
    
    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + GAMMA * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def record_obstacle(self, pos, action):
        self.obstacle_table.add_bad_action(pos, action)

# ============================================================
# 训练函数
# ============================================================
def train(env, agent, episodes=EPISODES, env_name='1'):
    print("="*60)
    print(f"训练 OREDQN - 环境 {env_name} (固定地图)")
    print(f"障碍物数量: {len(env.obstacles)}")
    print("="*60)
    
    total_steps = 0
    episode_rewards = []
    success_count = 0
    losses = []
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            action = agent.select_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            if info.get('reason') in ['collision', 'boundary']:
                agent.record_obstacle(env.agent_pos, action)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            total_steps += 1
            agent.steps_done = total_steps
            
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        if info.get('reason') == 'goal':
            success_count += 1
        
        if (episode+1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            success_rate = success_count / (episode+1) * 100
            elapsed = time.time() - start_time
            print(f"Ep {episode+1}/{episodes} | AvgRew: {avg_reward:6.1f} | "
                  f"SuccRate: {success_rate:5.1f}% | Eps: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")
    
    end_time = time.time()
    print("="*60)
    print(f"训练完成！总耗时: {end_time - start_time:.2f} 秒")
    print(f"总回合: {episodes}, 成功回合: {success_count}, 成功率: {success_count/episodes*100:.1f}%")
    print("="*60)
    return episode_rewards, losses

# ============================================================
# 测试与可视化
# ============================================================
def test(env, agent, num_episodes=3):
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        path = [env.agent_pos]
        while not done:
            action = agent.select_action(state, env)
            state, reward, done, info = env.step(action)
            path.append(env.agent_pos)
        fig, ax = plt.subplots(figsize=(7,7))
        env.render(path=path, ax=ax, title=f"Test Ep {ep+1} - {info.get('reason')}")
        plt.show()

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='1', choices=['1','2','3'],
                        help='选择环境: 1, 2, 3')
    parser.add_argument('--show_map', action='store_true',
                        help='仅显示地图，不训练')
    args = parser.parse_args()
    
    env = GridEnv(env_id=args.env)
    
    # 如果只是显示地图
    if args.show_map:
        env.show_map()
        print("地图已显示，对照论文图6检查是否一致。")
        print("如需调整障碍物，请修改 ENV1_OBSTACLES / ENV2_OBSTACLES / ENV3_OBSTACLES")
        exit()
    
    state_dim = 3 * GRID_SIZE * GRID_SIZE
    agent = OREDQNAgent(state_dim, ACTION_DIM)
    
    rewards, losses = train(env, agent, env_name=args.env)
    
    # 绘制奖励曲线
    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.6)
    window = 10
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smooth, 'r', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Reward Curve - Environment {args.env}')
    plt.grid(True)
    plt.show()
    
    test(env, agent, num_episodes=3)