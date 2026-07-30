import numpy as np
import random
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from dwa_env import DWAEnv

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DWA_PARAMS = {
    'v_max': 2.0,
    'w_max': 2.0,
    'v_acc': 1.0,
    'w_acc': 1.0,
    'dt': 0.1,
    'eval_time': 0.5,
}

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

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def train():
    print("=" * 60)
    print("Experiment 3: DQN + DWA (Original Reward)")
    print("Environment: env1 (15x15)")
    print("Reward: original distance-based")
    print("Motion: DWA continuous control")
    print("=" * 60)

    env = DWAEnv(dwa_params=DWA_PARAMS, coupled_reward=False)
    state_dim, action_dim = 11, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"DWA: 15x15, obstacles: {len(env.obstacles)}")
    print()

    EPISODES = 10000
    MAX_STEPS = 300
    BATCH_SIZE = 128
    GAMMA = 0.9
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.997
    TAU = 0.005
    MEMORY_CAPACITY = 20000

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(MEMORY_CAPACITY)

    epsilon = EPSILON_START
    success = 0
    episode_rewards, episode_lengths, episode_distances, episode_success = [], [], [], []

    start_time = time.time()
    print(f"Training {EPISODES} episodes...")

    for episode in range(EPISODES):
        state = env.reset()
        total_reward, done, step_count = 0, False, 0
        path = [env.agent_pos]

        while not done and step_count < MAX_STEPS:
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
            step_count += 1

            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))
                s_b = torch.FloatTensor(np.array(batch[0])).to(device)
                a_b = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
                ns_b = torch.FloatTensor(np.array(batch[2])).to(device)
                r_b = torch.FloatTensor(batch[3]).unsqueeze(1).to(device)
                d_b = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)
                current_q = policy_net(s_b).gather(1, a_b)
                with torch.no_grad():
                    next_q = target_net(ns_b).max(1, keepdim=True)[0]
                    target_q = r_b + (1 - d_b) * GAMMA * next_q
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            for tp, p in zip(target_net.parameters(), policy_net.parameters()):
                tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        final_dist = math.hypot(env.agent_pos[0] - env.goal[0], env.agent_pos[1] - env.goal[1])
        episode_distances.append(final_dist)
        episode_success.append(info.get('reason') == 'goal')
        if info.get('reason') == 'goal':
            success += 1

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            sr = success / (episode + 1) * 100
            elapsed = time.time() - start_time
            print(f"Ep {episode + 1}/{EPISODES} | AvgRew: {avg_reward:.2f} | SR: {sr:.1f}% | Eps: {epsilon:.3f} | Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    success_rate = success / EPISODES * 100
    print(f"\nComplete! Time: {total_time:.2f}s | SR: {success_rate:.1f}%")

    # Metrics
    sl = [episode_lengths[i] for i in range(EPISODES) if episode_success[i]]
    avg_sl = np.mean(sl) if sl else 0

    rc = EPISODES
    if len(episode_rewards) >= 200:
        fa = np.mean(episode_rewards[-100:])
        th = fa * 0.9
        for i in range(100, EPISODES - 50):
            if np.mean(episode_rewards[i:i+50]) >= th:
                rc = i + 50
                break

    dc = EPISODES
    if len(episode_distances) >= 200:
        for i in range(100, EPISODES - 50):
            if np.mean(episode_distances[i:i+50]) < 1.0:
                dc = i + 50
                break

    print(f"\n===== Metrics (DQN + DWA) =====")
    print(f"Time: {total_time:.2f}s | SR: {success_rate:.1f}%")
    print(f"Avg length: {avg_sl:.2f} | Reward conv: {rc}/{EPISODES} | Dist conv: {dc}/{EPISODES}")

    # Save model
    torch.save(policy_net.state_dict(), 'dqn_dwa_10000.pth')
    print(f"Model saved to dqn_dwa_10000.pth")

    return {'success_rate': success_rate, 'avg_length': avg_sl,
            'reward_conv': rc, 'dist_conv': dc, 'train_time': total_time}

if __name__ == "__main__":
    m = train()
    print(m)
