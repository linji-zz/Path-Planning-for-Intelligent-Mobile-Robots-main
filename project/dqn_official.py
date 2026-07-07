import numpy as np
import random
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =====================================================
# 环境 (8x8，无障碍物)
# =====================================================
GRID_SIZE = 8

class GridEnv:
    def __init__(self):
        self.size = GRID_SIZE
        self.actions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.action_dim = 8
        self.obstacles = set()
        self.start = (1,1)
        self.goal = (self.size-2, self.size-2)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        # 相对坐标 + 是否到达目标（辅助）
        dx = (self.goal[0] - self.agent_pos[0]) / self.size
        dy = (self.goal[1] - self.agent_pos[1]) / self.size
        return np.array([dx, dy, 0.0], dtype=np.float32)  # 第三维占位

    def step(self, action):
        self.steps += 1
        dx,dy = self.actions[action]
        old_pos = self.agent_pos
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            reward = -10.0
            self.done = True
            return self._get_state(), reward, True, {'reason':'boundary'}
        new_pos = (new_x, new_y)
        self.agent_pos = new_pos
        old_dist = math.hypot(old_pos[0]-self.goal[0], old_pos[1]-self.goal[1])
        new_dist = math.hypot(self.agent_pos[0]-self.goal[0], self.agent_pos[1]-self.goal[1])
        reward = (old_dist - new_dist) * 2.0
        if self.agent_pos == self.goal:
            reward = 50.0
            self.done = True
            return self._get_state(), reward, True, {'reason':'goal'}
        if self.steps >= 100:
            self.done = True
            return self._get_state(), reward, True, {'reason':'max_steps'}
        return self._get_state(), reward, False, {}

    def render(self, path=None, ax=None, title=""):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,4))
        ax.clear()
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        for i in range(self.size+1):
            ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
        if path:
            path_arr = np.array(path)
            if len(path_arr)>1:
                ax.plot(path_arr[:,1], path_arr[:,0], 'r-', linewidth=2)
            ax.plot(path_arr[-1,1], path_arr[-1,0], 'r^', markersize=8)
        ax.plot(self.start[1], self.start[0], 'bo', markersize=10)
        ax.plot(self.goal[1], self.goal[0], 'g*', markersize=12)
        ax.set_title(title)
        plt.draw()
        plt.pause(0.01)

# =====================================================
# DQN 网络 (官方风格)
# =====================================================
class DQN(nn.Module):
    def __init__(self, state_dim=3, action_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# =====================================================
# 经验回放 (官方风格)
# =====================================================
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# =====================================================
# 训练 (完全遵循官方DQN)
# =====================================================
def train():
    env = GridEnv()
    state_dim = 3
    action_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(10000)
    batch_size = 128
    gamma = 0.9
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 0.995  # 每回合衰减
    epsilon = epsilon_start

    EPISODES = 500
    success = 0
    rewards = []
    plt.ion()

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        path = [env.agent_pos]

        while not done:
            # 选择动作
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_t).max(1)[1].item()

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            path.append(env.agent_pos)

            # 存储经验
            memory.push((state, action, next_state, reward, done))
            state = next_state

            # 训练
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = list(zip(*transitions))
                state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
                action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
                next_state_batch = torch.FloatTensor(np.array(batch[2])).to(device)
                reward_batch = torch.FloatTensor(batch[3]).unsqueeze(1).to(device)
                done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

                current_q = policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    next_q = target_net(next_state_batch).max(1, keepdim=True)[0]
                    target_q = reward_batch + (1 - done_batch) * gamma * next_q

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 更新目标网络
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 衰减epsilon
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        rewards.append(total_reward)
        if info.get('reason') == 'goal':
            success += 1

        if (episode+1) % 20 == 0:
            avg = np.mean(rewards[-20:])
            sr = success/(episode+1)*100
            print(f"Ep {episode+1}/{EPISODES} | AvgRew: {avg:.2f} | SuccRate: {sr:.1f}% | Eps: {epsilon:.3f}")
            if (episode+1) % 50 == 0:
                fig, ax = plt.subplots(figsize=(4,4))
                env.render(path=path, ax=ax, title=f"Ep {episode+1}")
                plt.pause(0.5)
                plt.close()

    print(f"完成！成功率: {success}/{EPISODES} = {success/EPISODES*100:.1f}%")
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Official DQN - 8x8')
    plt.show()

if __name__ == "__main__":
    train()