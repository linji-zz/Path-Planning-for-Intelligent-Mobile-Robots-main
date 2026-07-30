"""
训练脚本 - DQN + 耦合奖励 + DWA 集成版

环境: env1 (15x15 网格地图)
奖励: 耦合奖励 = 距离奖励 + 方向对齐奖励(0.5) + 开阔度奖励(0.3)
运动: DWA 连续运动控制 (替代原始离散跳跃)

核心创新:
1. DQN 选择离散动作（8方向）-> 目标网格单元
2. DWA 规划平滑连续轨迹到达目标
3. 耦合奖励综合距离、方向、开阔度引导学习
"""

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


# DWA parameters for smooth motion in grid environment
DWA_PARAMS = {
    'v_max': 2.0,
    'w_max': 2.0,
    'v_acc': 1.0,
    'w_acc': 1.0,
    'dt': 0.1,
    'eval_time': 0.5,
}

# =====================================================
# DQN Network
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
# Replay Memory
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
# Training Function
# =====================================================
def train():
    print("=" * 60)
    print("DQN + Coupled Reward + DWA Integration")
    print("Environment: env1 (15x15)")
    print("Reward: distance + alignment(0.5) + openness(0.3)")
    print("Motion: DWA continuous control")
    print("=" * 60)

    env = DWAEnv(dwa_params=DWA_PARAMS, coupled_reward=True)
    state_dim = 11
    action_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"DWA Environment: 15x15, obstacles: {len(env.obstacles)}")
    print(f"DWA params: v_max={DWA_PARAMS['v_max']}, dt={DWA_PARAMS['dt']}")
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

    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    episode_success = []

    start_time = time.time()
    print(f"Training {EPISODES} episodes...")

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        path = [env.agent_pos]

        while not done and step_count < MAX_STEPS:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_t).max(1)[1].item()

            # DWAEnv.step() returns coupled reward directly
            # (distance_reward + alignment_reward + openness_reward)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            path.append(env.agent_pos)

            memory.push((state, action, next_state, reward, done))
            state = next_state
            step_count += 1

            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))
                state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
                action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
                next_state_batch = torch.FloatTensor(np.array(batch[2])).to(device)
                reward_batch = torch.FloatTensor(batch[3]).unsqueeze(1).to(device)
                done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

                current_q = policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    next_q = target_net(next_state_batch).max(1, keepdim=True)[0]
                    target_q = reward_batch + (1 - done_batch) * GAMMA * next_q

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Soft update target network
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        final_dist = math.hypot(env.agent_pos[0] - env.goal[0],
                                env.agent_pos[1] - env.goal[1])
        episode_distances.append(final_dist)

        if info.get('reason') == 'goal':
            success += 1
            episode_success.append(True)
        else:
            episode_success.append(False)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            sr = success / (episode + 1) * 100
            elapsed = time.time() - start_time
            print(f"Ep {episode + 1}/{EPISODES} | AvgRew: {avg_reward:.2f} | SR: {sr:.1f}% | Eps: {epsilon:.3f} | Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    success_rate = success / EPISODES * 100
    print(f"\nTraining complete! Total time: {total_time:.2f}s")
    print(f"Success rate: {success}/{EPISODES} = {success_rate:.1f}%")

    success_lengths = [episode_lengths[i] for i in range(EPISODES) if episode_success[i]]
    avg_success_length = np.mean(success_lengths) if success_lengths else 0

    # Reward convergence episode
    if len(episode_rewards) >= 200:
        final_avg = np.mean(episode_rewards[-100:])
        threshold = final_avg * 0.9
        reward_conv_episode = None
        for i in range(100, EPISODES - 50):
            if np.mean(episode_rewards[i:i + 50]) >= threshold:
                reward_conv_episode = i + 50
                break
        if reward_conv_episode is None:
            reward_conv_episode = EPISODES
    else:
        reward_conv_episode = EPISODES

    # Distance convergence episode
    dist_conv_episode = None
    if len(episode_distances) >= 200:
        for i in range(100, EPISODES - 50):
            if np.mean(episode_distances[i:i + 50]) < 1.0:
                dist_conv_episode = i + 50
                break
        if dist_conv_episode is None:
            dist_conv_episode = EPISODES
    else:
        dist_conv_episode = EPISODES

    print("\n===== Metrics Summary (DQN + Coupled Reward + DWA) =====")
    print(f"Training time: {total_time:.2f}s")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Avg path length (success): {avg_success_length:.2f} steps")
    print(f"Reward convergence: {reward_conv_episode}/{EPISODES}")
    print(f"Distance convergence: {dist_conv_episode}/{EPISODES}")

    # ===== Plotting =====
    window = 50
    episodes_range = np.arange(EPISODES)

    # Figure 1: Reward Curve
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, EPISODES), moving_avg, color='red', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Curve (DQN + Coupled Reward + DWA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig1_reward_dqn_dwa.png', dpi=150)
    plt.show()

    # Figure 2: Path Length
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_lengths, alpha=0.3, color='green', label='Episode Length')
    if len(episode_lengths) >= window:
        moving_avg_len = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, EPISODES), moving_avg_len, color='darkgreen', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Path Length (steps)')
    plt.title('Path Length Curve (DQN + Coupled Reward + DWA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_length_dqn_dwa.png', dpi=150)
    plt.show()

    # Figure 3: Distance to Goal
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_distances, alpha=0.3, color='purple', label='Distance to Goal')
    if len(episode_distances) >= window:
        moving_avg_dist = np.convolve(episode_distances, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, EPISODES), moving_avg_dist, color='magenta', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Distance Threshold (1.0)')
    plt.xlabel('Episode')
    plt.ylabel('Distance to Goal')
    plt.title('Distance Convergence (DQN + Coupled Reward + DWA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_distance_dqn_dwa.png', dpi=150)
    plt.show()

    # Figure 4: Success Rate
    plt.figure(figsize=(10, 6))
    cumulative_success = np.cumsum(episode_success) / (np.arange(EPISODES) + 1) * 100
    plt.plot(episodes_range, cumulative_success, color='orange', linewidth=2, label='Cumulative Success Rate')
    plt.axhline(y=success_rate, color='red', linestyle='--', linewidth=2,
                label=f'Final Success Rate ({success_rate:.1f}%)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title('Cumulative Success Rate (DQN + Coupled Reward + DWA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig4_success_dqn_dwa.png', dpi=150)
    plt.show()

    # Figure 5: Path Planning Result
    print("\nGenerating path planning result...")
    test_env = DWAEnv(dwa_params=DWA_PARAMS, coupled_reward=True)
    state = test_env.reset()
    done = False
    test_path = [test_env.agent_pos]
    policy_net.eval()
    with torch.no_grad():
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_t).argmax().item()
            state, reward, done, info = test_env.step(action)
            test_path.append(test_env.agent_pos)

    fig, ax = plt.subplots(figsize=(7, 7))
    test_env.render(path=test_path, ax=ax,
                    title=f"DQN + Coupled Reward + DWA - {info.get('reason')}")
    plt.tight_layout()
    plt.savefig('path_planning_dqn_dwa.png', dpi=150)
    plt.show()
    print("Path saved to path_planning_dqn_dwa.png")

    # Save model
    model_path = 'dqn_coupled_dwa_opt_10000.pth'
    torch.save(policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return {
        'success_rate': success_rate,
        'avg_length': avg_success_length,
        'reward_conv': reward_conv_episode,
        'dist_conv': dist_conv_episode,
        'train_time': total_time,
    }


if __name__ == "__main__":
    metrics = train()
    print(f"\nFinal metrics: {metrics}")
