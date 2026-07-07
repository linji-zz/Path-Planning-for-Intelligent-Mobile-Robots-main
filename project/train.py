import numpy as np
import torch
import random
import math
import time
from collections import deque
import matplotlib.pyplot as plt
import env
import dwa_planner
from agent import DQNAgent

GRID_SIZE = 15
EPISODES = 500
MAX_STEPS = 150
SEQ_LEN = 5
ACTION_DIM = 8

def main():
    robot_env = env.GridEnv(obstacle_ratio=0.08, seed=42)  # 降低障碍物
    dwa = dwa_planner.DWAPlanner()
    state_dim = 3 * GRID_SIZE * GRID_SIZE
    dqn_agent = DQNAgent(state_dim=state_dim, seq_len=SEQ_LEN, action_dim=ACTION_DIM)

    success_count = 0
    episode_rewards = []
    start_time = time.time()
    plt.ion()  # 交互模式

    for episode in range(EPISODES):
        state = robot_env.reset()
        flat_state = state.flatten()
        state_seq = deque([flat_state] * SEQ_LEN, maxlen=SEQ_LEN)
        total_reward = 0
        done = False
        step = 0
        path = [robot_env.agent_pos]

        while not done and step < MAX_STEPS:
            # DQN决策
            action = dqn_agent.select_action(list(state_seq))
            offsets = [(0,2),(2,0),(0,-2),(-2,0),(2,2),(2,-2),(-2,-2),(-2,2)]
            dx, dy = offsets[action]
            sub_x = robot_env.agent_pos[0] + dx
            sub_y = robot_env.agent_pos[1] + dy
            sub_x = max(0, min(GRID_SIZE-1, sub_x))
            sub_y = max(0, min(GRID_SIZE-1, sub_y))
            sub_goal = (sub_x, sub_y)

            # DWA执行
            v, w = dwa.plan(
                start_pos=robot_env.agent_pos,
                goal_pos=sub_goal,
                obstacles=robot_env.obstacles,
                current_v=0.0,
                current_w=0.0
            )
            # 速度转动作
            angle = math.atan2(w, v) if abs(v) > 0.01 else 0.0
            action_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
            closest_action = min(range(8), key=lambda i: abs(angle - action_angles[i]))
            next_state, reward, done, info = robot_env.step(closest_action)

            # 存储
            next_flat = next_state.flatten()
            next_state_seq = list(state_seq) + [next_flat]
            next_state_seq = next_state_seq[-SEQ_LEN:]
            dqn_agent.store_transition(list(state_seq), action, reward, next_state_seq, done)
            dqn_agent.train_step()

            state_seq.append(next_flat)
            total_reward += reward
            step += 1
            path.append(robot_env.agent_pos)

            if step % 10 == 0:
                dqn_agent.update_target()

            if done:
                break

        episode_rewards.append(total_reward)
        if info.get('reason') == 'goal':
            success_count += 1

        dqn_agent.decay_epsilon(episode, EPISODES)

        # 每10回合显示路径
        if (episode+1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            success_rate = success_count / (episode+1) * 100
            elapsed = time.time() - start_time
            print(f"Ep {episode+1}/{EPISODES} | AvgRew: {avg_reward:.2f} | SuccRate: {success_rate:.1f}% | Eps: {dqn_agent.epsilon:.3f} | Time: {elapsed:.0f}s")
            
            # 绘制当前路径
            fig, ax = plt.subplots(figsize=(6,6))
            robot_env.render(path=path, ax=ax, title=f"Episode {episode+1} - {info.get('reason')}")
            plt.pause(0.5)
            plt.close()

    print(f"\n训练完成！总耗时: {time.time()-start_time:.2f}s")
    print(f"成功率: {success_count}/{EPISODES} = {success_count/EPISODES*100:.1f}%")

    # 奖励曲线
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, alpha=0.6)
    if len(episode_rewards) >= 10:
        smooth = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(episode_rewards)), smooth, 'r', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()