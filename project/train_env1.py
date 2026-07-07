"""
升级版训练脚本 - 四图英文输出 + 图例说明
终端输出为中文
指标包括：训练时间、成功率、平均路径长度、奖励收敛轮数、距离收敛轮数
新增：训练完成后自动生成一张路径规划图
采用软更新目标网络（Polyak averaging）代替硬拷贝，提升训练稳定性
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
import os

# 导入环境1
from env1 import GridEnv

# 设置中文字体支持（仅用于路径显示，不影响英文图表）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']

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
# 训练函数
# =====================================================
def train():
    # 创建环境
    env = GridEnv()
    state_dim = 11
    action_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"环境: 15x15, 障碍物数量: {len(env.obstacles)}")

    # 超参数
    EPISODES = 10000
    MAX_STEPS = 300
    BATCH_SIZE = 128
    GAMMA = 0.9
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.997
    TAU = 0.005          # 软更新系数（Polyak 平均）
    MEMORY_CAPACITY = 20000

    # 网络和优化器
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(MEMORY_CAPACITY)

    epsilon = EPSILON_START
    success = 0

    # ===== 记录指标 =====
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    episode_success = []
    losses = []

    start_time = time.time()

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
                losses.append(loss.item())

        # ===== 软更新目标网络（Polyak 平均） =====
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

        if (episode+1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            sr = success / (episode+1) * 100
            elapsed = time.time() - start_time
            print(f"回合 {episode+1}/{EPISODES} | 平均奖励: {avg_reward:.2f} | 成功率: {sr:.1f}% | 探索率: {epsilon:.3f} | 耗时: {elapsed:.0f}s")

    # 训练结束
    total_time = time.time() - start_time
    success_rate = success / EPISODES * 100
    print(f"\n训练完成！总耗时: {total_time:.2f} 秒")
    print(f"成功率: {success}/{EPISODES} = {success_rate:.1f}%")

    # ===== 计算各项指标 =====
    success_lengths = [episode_lengths[i] for i in range(EPISODES) if episode_success[i]]
    avg_success_length = np.mean(success_lengths) if success_lengths else 0

    # 奖励收敛轮数
    if len(episode_rewards) >= 200:
        final_avg = np.mean(episode_rewards[-100:])
        threshold = final_avg * 0.9
        reward_conv_episode = None
        for i in range(100, EPISODES - 50):
            if np.mean(episode_rewards[i:i+50]) >= threshold:
                reward_conv_episode = i + 50
                break
        if reward_conv_episode is None:
            reward_conv_episode = EPISODES
    else:
        reward_conv_episode = EPISODES

    # 距离收敛轮数
    dist_conv_episode = None
    if len(episode_distances) >= 200:
        for i in range(100, EPISODES - 50):
            if np.mean(episode_distances[i:i+50]) < 1.0:
                dist_conv_episode = i + 50
                break
        if dist_conv_episode is None:
            dist_conv_episode = EPISODES
    else:
        dist_conv_episode = EPISODES

    # ===== 中文终端输出 =====
    print("\n===== 指标汇总 =====")
    print(f"训练时间: {total_time:.2f} 秒")
    print(f"成功率: {success_rate:.1f}%")
    print(f"平均路径长度（成功回合）: {avg_success_length:.2f} 步")
    print(f"奖励收敛轮数: {reward_conv_episode}/{EPISODES}")
    print(f"距离收敛轮数: {dist_conv_episode}/{EPISODES}")

    # =====================================================
    # 绘制四张独立图表（英文 + 图例）
    # =====================================================
    window = 50
    episodes_range = np.arange(EPISODES)

    # ---- 图1: Reward Curve ----
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, EPISODES), moving_avg, color='red', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Reward Curve', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig1_reward_curve.png', dpi=150)
    plt.show()

    # ---- 图2: Path Length Curve ----
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_lengths, alpha=0.3, color='green', label='Episode Length')
    if len(episode_lengths) >= window:
        moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, EPISODES), moving_avg_len, color='darkgreen', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Path Length (steps)', fontsize=12)
    plt.title('Path Length Curve', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_path_length_curve.png', dpi=150)
    plt.show()

    # ---- 图3: Distance to Goal Curve ----
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_distances, alpha=0.3, color='purple', label='Distance to Goal')
    if len(episode_distances) >= window:
        moving_avg_dist = np.convolve(episode_distances, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, EPISODES), moving_avg_dist, color='magenta', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Distance Threshold (1.0)')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Distance to Goal', fontsize=12)
    plt.title('Distance Convergence Curve', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_distance_curve.png', dpi=150)
    plt.show()

    # ---- 图4: Cumulative Success Rate ----
    plt.figure(figsize=(10, 6))
    cumulative_success = np.cumsum(episode_success) / (np.arange(EPISODES) + 1) * 100
    plt.plot(episodes_range, cumulative_success, color='orange', linewidth=2,
             label='Cumulative Success Rate')
    plt.axhline(y=success_rate, color='red', linestyle='--', linewidth=2,
                label=f'Final Success Rate ({success_rate:.1f}%)')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Cumulative Success Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig4_success_rate.png', dpi=150)
    plt.show()

    # =====================================================
    # 生成一张路径规划图（使用训练好的策略）
    # =====================================================
    print("\n正在生成路径规划图...")
    test_env = GridEnv()
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
    test_env.render(path=test_path, ax=ax, title=f"Path Planning Result - {info.get('reason')}")
    plt.tight_layout()
    plt.savefig('path_planning_result.png', dpi=150)
    plt.show()
    print(f"路径规划图已保存为 path_planning_result.png")

    # 保存模型
    model_path = 'dqn_env1_baseline_10000.pth'
    torch.save(policy_net.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")

    return {
        'success_rate': success_rate,
        'avg_length': avg_success_length,
        'reward_conv': reward_conv_episode,
        'dist_conv': dist_conv_episode,
        'train_time': total_time,
    }

# =====================================================
# 主程序
# =====================================================
if __name__ == "__main__":
    metrics = train()