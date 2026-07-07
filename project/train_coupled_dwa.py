"""
训练脚本 - DWA + 耦合奖励（消融实验 Exp 4：原始权重版）
环境：env1（基于 env1 地图）
奖励：距离奖励 + 方向对齐奖励(0.5) + 开阔度奖励(0.3)
图片只弹窗，不保存到项目目录
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

from env1 import GridEnv

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
# 辅助函数：计算局部开阔度
# =====================================================
def local_openness(env, target_x, target_y):
    """计算子目标周围 3x3 邻域内障碍物占比"""
    obstacle_count = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = target_x + dx, target_y + dy
            if 0 <= nx < env.size and 0 <= ny < env.size:
                if (nx, ny) in env.obstacles:
                    obstacle_count += 1
    return 1.0 - obstacle_count / 9.0

# =====================================================
# 训练函数
# =====================================================
def train():
    print("=" * 60)
    print("实验: DQN + DWA + 耦合奖励（原始权重版）")
    print("方向对齐: 0.5, 开阔度: 0.3")
    print("=" * 60)

    env = GridEnv()
    state_dim = 11
    action_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"环境: 15x15, 障碍物数量: {len(env.obstacles)}")
    print()

    # 超参数
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

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        path = [env.agent_pos]
        agent_theta = 0.0

        while not done and step_count < MAX_STEPS:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_t).max(1)[1].item()

            old_pos = env.agent_pos
            dx, dy = env.actions[action]
            sub_goal_x = old_pos[0] + dx
            sub_goal_y = old_pos[1] + dy

            # ===== 执行一步 =====
            next_state, reward_original, done, info = env.step(action)

            # ===== 计算耦合奖励（原始权重：0.5 和 0.3） =====
            # 1. 距离奖励
            old_dist = math.hypot(old_pos[0] - env.goal[0], old_pos[1] - env.goal[1])
            new_dist = math.hypot(env.agent_pos[0] - env.goal[0], env.agent_pos[1] - env.goal[1])
            reward_dist = (old_dist - new_dist) * 2.0

            # 2. 方向对齐奖励（原始权重 0.5）
            angle_to_subgoal = math.atan2(sub_goal_y - env.agent_pos[1],
                                          sub_goal_x - env.agent_pos[0])
            alignment_reward = 0.5 * math.cos(agent_theta - angle_to_subgoal)

            # 3. 局部开阔度奖励（原始权重 0.3）
            openness = local_openness(env, int(round(sub_goal_x)), int(round(sub_goal_y)))
            openness_reward = 0.3 * openness

            # 4. 合成奖励
            reward = reward_dist + alignment_reward + openness_reward

            # 5. 碰撞和终点奖励覆盖
            if done and info.get('reason') == 'goal':
                reward = 50.0
            elif done and info.get('reason') in ['collision', 'boundary']:
                reward = -10.0

            # 6. 更新朝向角
            if new_dist < old_dist:
                move_angle = math.atan2(env.agent_pos[1] - old_pos[1],
                                        env.agent_pos[0] - old_pos[0])
                agent_theta = move_angle

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

        # 软更新
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
            print(f"回合 {episode + 1}/{EPISODES} | 平均奖励: {avg_reward:.2f} | 成功率: {sr:.1f}% | 探索率: {epsilon:.3f} | 耗时: {elapsed:.0f}s")

    total_time = time.time() - start_time
    success_rate = success / EPISODES * 100
    print(f"\n训练完成！总耗时: {total_time:.2f} 秒")
    print(f"成功率: {success}/{EPISODES} = {success_rate:.1f}%")

    # 计算指标
    success_lengths = [episode_lengths[i] for i in range(EPISODES) if episode_success[i]]
    avg_success_length = np.mean(success_lengths) if success_lengths else 0

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

    print("\n===== 指标汇总（DQN + DWA + 耦合奖励 原始权重） =====")
    print(f"训练时间: {total_time:.2f} 秒")
    print(f"成功率: {success_rate:.1f}%")
    print(f"平均路径长度（成功回合）: {avg_success_length:.2f} 步")
    print(f"奖励收敛轮数: {reward_conv_episode}/{EPISODES}")
    print(f"距离收敛轮数: {dist_conv_episode}/{EPISODES}")

    # ===== 绘图（只弹窗，不保存） =====
    window = 50
    episodes_range = np.arange(EPISODES)

    # 图1: Reward Curve
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, EPISODES), moving_avg, color='red', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Curve (DQN + DWA + Coupled Reward - Original Weights)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(...) 已移除
    plt.show()

    # 图2: Path Length
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_lengths, alpha=0.3, color='green', label='Episode Length')
    if len(episode_lengths) >= window:
        moving_avg_len = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, EPISODES), moving_avg_len, color='darkgreen', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Path Length (steps)')
    plt.title('Path Length Curve (DQN + DWA + Coupled Reward - Original Weights)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(...) 已移除
    plt.show()

    # 图3: Distance
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, episode_distances, alpha=0.3, color='purple', label='Distance to Goal')
    if len(episode_distances) >= window:
        moving_avg_dist = np.convolve(episode_distances, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, EPISODES), moving_avg_dist, color='magenta', linewidth=2,
                 label=f'Moving Average (window={window})')
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Distance Threshold (1.0)')
    plt.xlabel('Episode')
    plt.ylabel('Distance to Goal')
    plt.title('Distance Convergence (DQN + DWA + Coupled Reward - Original Weights)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(...) 已移除
    plt.show()

    # 图4: Success Rate
    plt.figure(figsize=(10, 6))
    cumulative_success = np.cumsum(episode_success) / (np.arange(EPISODES) + 1) * 100
    plt.plot(episodes_range, cumulative_success, color='orange', linewidth=2, label='Cumulative Success Rate')
    plt.axhline(y=success_rate, color='red', linestyle='--', linewidth=2,
                label=f'Final Success Rate ({success_rate:.1f}%)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title('Cumulative Success Rate (DQN + DWA + Coupled Reward - Original Weights)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(...) 已移除
    plt.show()

    # 图5: 路径规划图
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
    test_env.render(path=test_path, ax=ax, title=f"DQN + DWA + Coupled Reward (Original Weights) - {info.get('reason')}")
    plt.tight_layout()
    # plt.savefig(...) 已移除
    plt.show()
    print("路径规划图已显示（未保存到本地）")

    torch.save(policy_net.state_dict(), 'dqn_coupled_dwa_10000.pth')
    print("模型已保存为 dqn_coupled_dwa_10000.pth")

    return {
        'success_rate': success_rate,
        'avg_length': avg_success_length,
        'reward_conv': reward_conv_episode,
        'dist_conv': dist_conv_episode,
        'train_time': total_time,
    }

if __name__ == "__main__":
    metrics = train()