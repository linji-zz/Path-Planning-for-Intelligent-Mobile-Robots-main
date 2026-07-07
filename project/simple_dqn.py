import numpy as np
import random
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ===== 环境 =====
GRID_SIZE = 15

class GridEnv:
    def __init__(self, obstacle_ratio=0.08, seed=42):
        self.size = GRID_SIZE
        self.actions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.action_dim = 8
        random.seed(seed)
        np.random.seed(seed)
        self.obstacles = self._gen_obstacles(obstacle_ratio)
        self.start = (1,1)
        self.goal = (self.size-2, self.size-2)
        while self.start in self.obstacles:
            self.start = (random.randint(1,self.size-2), random.randint(1,self.size-2))
        while self.goal in self.obstacles or self.goal == self.start:
            self.goal = (random.randint(1,self.size-2), random.randint(1,self.size-2))
        self.reset()

    def _gen_obstacles(self, ratio):
        obs = set()
        num = int(self.size*self.size*ratio)
        while len(obs) < num:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if (x,y) not in [(1,1), (self.size-2, self.size-2)]:
                obs.add((x,y))
        return obs

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        x,y = self.agent_pos
        state[0,x,y] = 1.0
        for (ox,oy) in self.obstacles:
            state[1,ox,oy] = 1.0
        gx,gy = self.goal
        state[2,gx,gy] = 1.0
        return state

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
        if new_pos in self.obstacles:
            reward = -10.0
            self.done = True
            return self._get_state(), reward, True, {'reason':'collision'}
        self.agent_pos = new_pos
        old_dist = math.hypot(old_pos[0]-self.goal[0], old_pos[1]-self.goal[1])
        new_dist = math.hypot(self.agent_pos[0]-self.goal[0], self.agent_pos[1]-self.goal[1])
        reward = (old_dist - new_dist) * 0.5
        if new_dist < old_dist:
            reward += 0.3
        else:
            reward -= 0.2
        if self.agent_pos == self.goal:
            reward = 20.0
            self.done = True
            return self._get_state(), reward, True, {'reason':'goal'}
        if self.steps >= 150:
            self.done = True
            return self._get_state(), reward, True, {'reason':'max_steps'}
        return self._get_state(), reward, False, {}

    def render(self, path=None, ax=None, title=""):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
        ax.clear()
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        for i in range(self.size+1):
            ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
        for (ox,oy) in self.obstacles:
            rect = Rectangle((ox-0.5, oy-0.5), 1, 1, facecolor='black', edgecolor='none')
            ax.add_patch(rect)
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

# ===== DQN网络 =====
class DQN(nn.Module):
    def __init__(self, input_dim=3*GRID_SIZE*GRID_SIZE, output_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ===== 智能体 =====
class Agent:
    def __init__(self, state_dim, action_dim=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DQN(state_dim, action_dim).to(self.device)
        self.target = DQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 0.9
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
            q = self.policy(state_t)
            return q.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array([s.flatten() for s in states])).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([s.flatten() for s in next_states])).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q = self.policy(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target(next_states).max(1, keepdim=True)[0]
            target = rewards + (1-dones) * self.gamma * max_next_q
        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def decay_epsilon(self, episode, total):
        self.epsilon = max(0.1, 0.9 - episode/total*0.8)

# ===== 训练 =====
def train():
    env = GridEnv()
    agent = Agent(3*GRID_SIZE*GRID_SIZE)
    EPISODES = 500
    success = 0
    rewards = []
    plt.ion()
    for ep in range(EPISODES):
        state = env.reset()
        path = [env.agent_pos]
        total_r = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_r += reward
            path.append(env.agent_pos)
            if done:
                break
        rewards.append(total_r)
        if info.get('reason') == 'goal':
            success += 1
        agent.decay_epsilon(ep, EPISODES)
        if (ep+1) % 20 == 0:
            avg = np.mean(rewards[-20:])
            sr = success/(ep+1)*100
            print(f"Ep {ep+1}/{EPISODES} | AvgRew: {avg:.2f} | SuccRate: {sr:.1f}% | Eps: {agent.epsilon:.3f}")
            if (ep+1) % 50 == 0:
                fig, ax = plt.subplots(figsize=(5,5))
                env.render(path=path, ax=ax, title=f"Ep {ep+1}")
                plt.pause(0.5)
                plt.close()
        if (ep+1) % 100 == 0:
            agent.update_target()
    print(f"完成！成功率: {success}/{EPISODES} = {success/EPISODES*100:.1f}%")
    plt.figure()
    plt.plot(rewards)
    plt.show()

if __name__ == "__main__":
    train()