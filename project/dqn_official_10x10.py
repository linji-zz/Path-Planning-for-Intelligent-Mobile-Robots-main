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
# DWA 规划器 (轻量级)
# =====================================================
class DWAPlanner:
    def __init__(self, v_max=0.8, w_max=0.8, dt=0.1, eval_time=0.5):
        self.v_max = v_max
        self.w_max = w_max
        self.dt = dt
        self.eval_time = eval_time
        self.sample_num = 8

    def plan(self, start, goal, obstacles, current_v=0.0, current_w=0.0):
        # 速度采样
        v_samples = np.linspace(0.1, self.v_max, self.sample_num)
        w_samples = np.linspace(-self.w_max, self.w_max, self.sample_num)
        best_score = -float('inf')
        best_v, best_w = 0.1, 0.0
        for v in v_samples:
            for w in w_samples:
                traj = self._simulate(start, v, w)
                if self._collide(traj, obstacles):
                    continue
                score = self._evaluate(traj, goal, obstacles, v)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
        return best_v, best_w

    def _simulate(self, start, v, w):
        traj = []
        x, y = start
        theta = 0.0
        steps = int(self.eval_time / self.dt)
        for _ in range(steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += w * self.dt
            traj.append((x, y))
        return traj

    def _collide(self, traj, obstacles):
        for x, y in traj[::2]:
            if not (0 <= x < 10 and 0 <= y < 10):
                return True
            if (int(round(x)), int(round(y))) in obstacles:
                return True
        return False

    def _evaluate(self, traj, goal, obstacles, v):
        if not traj:
            return 0.0
        fx, fy = traj[-1]
        gx, gy = goal
        dist = math.hypot(fx-gx, fy-gy)
        heading = 1.0 / (1.0 + dist) if dist > 0 else 1.0
        clearance = 1.0
        for x, y in traj:
            min_d = min([math.hypot(x-ox, y-oy) for ox, oy in obstacles] or [10])
            clearance = min(clearance, min_d / 2.0)
        clearance = min(clearance, 1.0)
        return 0.5 * heading + 0.3 * clearance + 0.2 * (v / self.v_max)

# =====================================================
# 环境 (10x10，含障碍物感知，集成DWA)
# =====================================================
GRID_SIZE = 10

class GridEnv:
    def __init__(self, obstacle_ratio=0.05, seed=42):
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
        self.dwa = DWAPlanner()
        self.reset()

    def _gen_obstacles(self, ratio):
        obs = set()
        num = int(self.size*self.size*ratio)
        attempts = 0
        while len(obs) < num and attempts < 10000:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if (x,y) not in [(1,1), (self.size-2, self.size-2)]:
                obs.add((x,y))
            attempts += 1
        return obs

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        dx = (self.goal[0] - self.agent_pos[0]) / self.size
        dy = (self.goal[1] - self.agent_pos[1]) / self.size
        dist = math.hypot(dx*self.size, dy*self.size) / self.size
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
        # ===== DWA 执行：向动作方向平滑移动 =====
        dx, dy = self.actions[action]
        sub_goal = (self.agent_pos[0] + dx*2, self.agent_pos[1] + dy*2)
        sub_goal = (max(0, min(self.size-1, sub_goal[0])), max(0, min(self.size-1, sub_goal[1])))
        
        # DWA规划
        v, w = self.dwa.plan(self.agent_pos, sub_goal, self.obstacles)
        # 将速度转换回离散动作（取最近方向）
        if abs(v) < 0.01:
            angle = 0.0
        else:
            angle = math.atan2(w, v) if v > 0 else 0.0
        action_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        dwa_action = min(range(8), key=lambda i: abs(angle - action_angles[i]))
        
        # 执行动作（DWA选择的动作，但本质上与原始动作不同，这里用原始动作执行以便训练）
        # 实际上，我们直接用原始动作，但为了体现DWA，我们使用dwa_action
        # 但为了训练稳定，这里我们用原始action，DWA仅用于平滑（不改变方向）
        # 更安全的做法：直接用原始动作走网格，但增加随机性模拟平滑
        # 这里我们混合：50%概率走原始动作，50%走DWA优化方向
        if random.random() < 0.5:
            final_action = action
        else:
            final_action = dwa_action
        
        dx2, dy2 = self.actions[final_action]
        old_pos = self.agent_pos
        new_x = old_pos[0] + dx2
        new_y = old_pos[1] + dy2
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
        reward = (old_dist - new_dist) * 2.0
        if self.agent_pos == self.goal:
            reward = 50.0
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

# =====================================================
# DQN 网络
# =====================================================
class DQN(nn.Module):
    def __init__(self, state_dim=11, action_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# =====================================================
# 经验回放
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
# 训练
# =====================================================
def train():
    env = GridEnv(obstacle_ratio=0.05, seed=42)
    state_dim = 11
    action_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(20000)
    batch_size = 128
    gamma = 0.9
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start

    EPISODES = 600
    success = 0
    rewards = []
    plt.ion()

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        path = [env.agent_pos]

        while not done:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_t).max(1)[1].item()

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            path.append(env.agent_pos)

            memory.push((state, action, next_state, reward, done))
            state = next_state

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

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

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
                fig, ax = plt.subplots(figsize=(5,5))
                env.render(path=path, ax=ax, title=f"Ep {episode+1} (DWA平滑)")
                plt.pause(0.5)
                plt.close()

    print(f"完成！成功率: {success}/{EPISODES} = {success/EPISODES*100:.1f}%")
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN + DWA Smooth - 10x10')
    plt.show()

if __name__ == "__main__":
    train()