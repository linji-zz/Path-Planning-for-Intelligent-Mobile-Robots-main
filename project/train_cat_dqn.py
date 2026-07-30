"""train_cat_dqn.py - Cross-Attention Transformer DQN + 耦合奖励"""
import sys, numpy as np, random, math, time
from collections import deque
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")
from env1 import GridEnv
from models import CrossAttentionDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

env = GridEnv()
SEQ_LEN = 5
state_dim, action_dim = 11, 8

policy = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)
target = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)
target.load_state_dict(policy.state_dict())
target.eval()
opt = optim.Adam(policy.parameters(), lr=0.001)
memory = deque(maxlen=20000)

def local_openness(env, tx, ty):
    cnt = 0
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            nx, ny = tx+dx, ty+dy
            if 0 <= nx < env.size and 0 <= ny < env.size and (nx,ny) in env.obstacles:
                cnt += 1
    return 1.0 - cnt/9.0

def to_seq(history):
    return np.stack(list(history)).astype(np.float32)  # (seq_len, 11)

EPISODES = 500
MAX_STEPS = 300
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START, EPS_END, EPS_DECAY = 0.9, 0.05, 0.997
TAU = 0.005

eps = EPS_START
successes, rewards = [], []
start_t = time.time()

for ep in range(EPISODES):
    s = env.reset()
    history = deque([s] * SEQ_LEN, maxlen=SEQ_LEN)  # 填充历史
    tr, done, steps = 0, False, 0
    agent_theta = 0.0

    while not done and steps < MAX_STEPS:
        if random.random() < eps:
            action = random.randrange(action_dim)
        else:
            with torch.no_grad():
                seq = torch.FloatTensor(to_seq(history)).unsqueeze(0).to(device)
                action = policy(seq).max(1)[1].item()

        old_pos = env.agent_pos
        dx, dy = env.actions[action]
        sub_goal = (old_pos[0]+dx, old_pos[1]+dy)

        ns, r_orig, done, info = env.step(action)

        # 耦合奖励
        old_dist = math.hypot(old_pos[0]-env.goal[0], old_pos[1]-env.goal[1])
        new_dist = math.hypot(env.agent_pos[0]-env.goal[0], env.agent_pos[1]-env.goal[1])
        r_dist = (old_dist-new_dist)*2.0

        angle_to_sub = math.atan2(sub_goal[1]-env.agent_pos[1], sub_goal[0]-env.agent_pos[0])
        r_align = 0.5 * math.cos(agent_theta - angle_to_sub)

        openness = local_openness(env, int(round(sub_goal[0])), int(round(sub_goal[1])))
        r_open = 0.3 * openness

        reward = r_dist + r_align + r_open
        if done:
            if info.get("reason") == "goal": reward = 50.0
            elif info.get("reason") in ("collision","boundary"): reward = -10.0

        if new_dist < old_dist:
            move_angle = math.atan2(env.agent_pos[1]-old_pos[1], env.agent_pos[0]-old_pos[0])
            agent_theta = move_angle

        tr += reward
        steps += 1

        # 构建下一步的历史序列
        next_history = list(history)[1:] + [ns]
        memory.append((to_seq(history), action, np.array(next_history, dtype=np.float32), reward, done))
        history.append(ns)

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            b = list(zip(*batch))
            sb = torch.FloatTensor(np.array(b[0])).to(device)
            ab = torch.LongTensor(b[1]).unsqueeze(1).to(device)
            nsb = torch.FloatTensor(np.array(b[2])).to(device)
            rb = torch.FloatTensor(b[3]).unsqueeze(1).to(device)
            db = torch.FloatTensor(b[4]).unsqueeze(1).to(device)

            curr = policy(sb).gather(1, ab)
            with torch.no_grad():
                tgt = rb + (1-db)*GAMMA*target(nsb).max(1,keepdim=True)[0]
            nn.MSELoss()(curr, tgt).backward()
            opt.step()
            opt.zero_grad()

    with torch.no_grad():
        for tp, p in zip(target.parameters(), policy.parameters()):
            tp.data.copy_(TAU*p.data+(1-TAU)*tp.data)
    if eps > EPS_END: eps *= EPS_DECAY

    rewards.append(tr)
    succ = info.get("reason")=="goal"
    successes.append(succ)

    if (ep+1)%100==0:
        avg_r = np.mean(rewards[-100:])
        sr = sum(successes[-100:])
        print(f"Ep {ep+1}/{EPISODES} | AvgRew: {avg_r:.1f} | SR: {sr:.0f}% | Eps: {eps:.3f} | Time: {time.time()-start_t:.0f}s", flush=True)

t = time.time()-start_t
print(f"\nDone! Time: {t:.0f}s", flush=True)

# 评估
print("\n===== Greedy Eval (200 episodes) =====", flush=True)
policy.eval()
succ_g, len_g = 0, []
for ep in range(200):
    s = env.reset()
    history = deque([s]*SEQ_LEN, maxlen=SEQ_LEN)
    done, steps = False, 0
    while not done and steps < MAX_STEPS:
        with torch.no_grad():
            seq = torch.FloatTensor(to_seq(history)).unsqueeze(0).to(device)
            a = policy(seq).max(1)[1].item()
        ns, r, done, info = env.step(a)
        history.append(ns)
        steps += 1
    if info.get("reason")=="goal":
        succ_g += 1
        len_g.append(steps)

avg_len = np.mean(len_g) if len_g else 0
print(f"Greedy: {succ_g}/200 = {succ_g/2:.1f}% | Avg success len: {avg_len:.1f}", flush=True)
torch.save(policy.state_dict(), "dqn_cat_500.pth")
print("Model saved.", flush=True)

