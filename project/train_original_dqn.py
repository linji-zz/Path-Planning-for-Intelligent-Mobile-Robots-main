"""
原始 DQN 训练脚本（修正版）- 瞬移环境
用 Adam + 更保守的超参数
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

from env_original import GridEnv

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
    env = GridEnv()
    state_dim = 11
    action_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"环境: 15x15, 障碍物数量: {len(env.obstacles)}")

    # ========== 修正后的超参数 ==========
    EPISODES = 10000
    MAX_STEPS = 300
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.05                 # 从0.1降到0.05
    EPSILON_DECAY = 0.997              # 更慢衰减
    TARGET_UPDATE = 500                # 从1000降到500（更温和）
    MEMORY_CAPACITY = 20000
    LEARNING_RATE = 0.001              # 改用Adam，学习率更高
    MIN_MEMORY = 500                   # 先积累500个样本再训练

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)  # 换Adam
    memory = ReplayMemory(MEMORY_CAPACITY)

    epsilon = EPSILON_START
    success = 0

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

        while not done and step_count < MAX_STEPS:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_t).max(1)[1].item()

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            memory.push((state, action, next_state, reward, done))
            state = next_state
            step_count += 1

            # ===== 关键修复：积累足够样本再训练 =====
            if len(memory) >= MIN_MEMORY:
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

        # ===== 硬更新 =====
        if episode % TARGET_UPDATE == 0 and episode > 0:
            target_net.load_state_dict(policy_net.state_dict())

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

    total_time = time.time() - start_time
    success_rate = success / EPISODES * 100
    print(f"\n训练完成！总耗时: {total_time:.2f} 秒")
    print(f"成功率: {success}/{EPISODES} = {success_rate:.1f}%")

    # 收敛轮数计算
    success_lengths = [episode_lengths[i] for i in range(EPISODES) if episode_success[i]]
    avg_success_length = np.mean(success_lengths) if success_lengths else 0

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

    print("\n===== 指标汇总（原始 DQN + 瞬移环境 修正版） =====")
    print(f"训练时间: {total_time:.2f} 秒")
    print(f"成功率: {success_rate:.1f}%")
    print(f"平均路径长度（成功回合）: {avg_success_length:.2f} 步")
    print(f"奖励收敛轮数: {reward_conv_episode}/{EPISODES}")
    print(f"距离收敛轮数: {dist_conv_episode}/{EPISODES}")

    # 绘制曲线
    window = 50
    episodes_range = np.arange(EPISODES)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, EPISODES), moving_avg, color='red', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Curve (Original DQN + Discrete Jump - Fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig1_reward_original_fixed.png', dpi=150)
    plt.show()

    # 路径图
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
    test_env.render(path=test_path, ax=ax, title=f"Original DQN Path (Fixed) - {info.get('reason')}")
    plt.tight_layout()
    plt.savefig('path_planning_result_original_fixed.png', dpi=150)
    plt.show()

    torch.save(policy_net.state_dict(), 'dqn_original_fixed_10000.pth')
    print("模型已保存为 dqn_original_fixed_10000.pth")

    return {
        'success_rate': success_rate,
        'avg_length': avg_success_length,
        'reward_conv': reward_conv_episode,
        'dist_conv': dist_conv_episode,
        'train_time': total_time,
    }

if __name__ == "__main__":
    metrics = train()