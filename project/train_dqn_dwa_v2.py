import sys
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")
"""train_dqn_dwa_v2.py - DQN + DWA (FIXED VERSION)
DWA fixes:
1. heading = distance improvement (start_dist - end_dist)
2. current_theta passed to planner
3. full eval_time/dt sub-steps executed
4. obstacle-safe sub-goal clamping
5. full velocity range sampling [0, v_max]
6. all trajectory points checked for collision
"""
import numpy as np, random, math, time
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

from dwa_env import DWAEnv

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DWA_PARAMS = {
    "v_max": 2.0, "w_max": 2.0, "v_acc": 1.0, "w_acc": 1.0,
    "dt": 0.1, "eval_time": 0.5,
}

class DQN(nn.Module):
    def __init__(self, state_dim=11, action_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)

class ReplayMemory:
    def __init__(self, cap): self.memory = deque(maxlen=cap)
    def push(self, t): self.memory.append(t)
    def sample(self, bs): return random.sample(self.memory, bs)
    def __len__(self): return len(self.memory)

def train():
    EPISODES = 500  # Quick validation
    MAX_STEPS = 200
    BATCH_SIZE = 128
    GAMMA = 0.9
    EPS_START, EPS_END, EPS_DECAY = 0.9, 0.05, 0.997
    TAU, MEM_CAP = 0.005, 20000

    env = DWAEnv(dwa_params=DWA_PARAMS, coupled_reward=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    policy = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    opt = optim.Adam(policy.parameters(), lr=0.001)
    mem = ReplayMemory(MEM_CAP)

    eps = EPS_START
    successes, rewards, lengths = [], [], []

    start_t = time.time()
    for ep in range(EPISODES):
        s = env.reset()
        tr, done, steps = 0, False, 0

        while not done and steps < MAX_STEPS:
            if random.random() < eps:
                a = random.randrange(8)
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(s).unsqueeze(0).to(device)
                    a = policy(st).max(1)[1].item()
            ns, r, done, info = env.step(a)
            tr += r
            mem.push((s, a, ns, r, done))
            s = ns
            steps += 1

            if len(mem) >= BATCH_SIZE:
                batch = list(zip(*mem.sample(BATCH_SIZE)))
                sb = torch.FloatTensor(np.array(batch[0])).to(device)
                ab = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
                nsb = torch.FloatTensor(np.array(batch[2])).to(device)
                rb = torch.FloatTensor(batch[3]).unsqueeze(1).to(device)
                db = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

                curr = policy(sb).gather(1, ab)
                with torch.no_grad():
                    nxt = target(nsb).max(1, keepdim=True)[0]
                    tgt = rb + (1-db) * GAMMA * nxt
                loss = nn.MSELoss()(curr, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            for tp, p in zip(target.parameters(), policy.parameters()):
                tp.data.copy_(TAU * p.data + (1-TAU) * tp.data)

        if eps > EPS_END: eps *= EPS_DECAY

        rewards.append(tr)
        lengths.append(steps)
        is_succ = info.get("reason") == "goal"
        successes.append(is_succ)

        if (ep+1) % 50 == 0:
            avg_r = np.mean(rewards[-50:])
            sr = sum(successes)/(ep+1)*100
            el = time.time()-start_t
            print(f"Ep {ep+1}/{EPISODES} | AvgRew: {avg_r:.2f} | SR: {sr:.1f}% | Eps: {eps:.3f} | Time: {el:.0f}s")

    t = time.time()-start_t
    sr = sum(successes)/EPISODES*100
    print(f"\nDone! Time: {t:.2f}s | SR: {sr:.1f}%")

    # Evaluate
    print("\n--- Evaluation (100 episodes) ---")
    policy.eval()
    eval_succ = 0
    for ep in range(100):
        s = env.reset()
        d, st = False, 0
        while not d and st < MAX_STEPS:
            with torch.no_grad():
                st_t = torch.FloatTensor(s).unsqueeze(0).to(device)
                a = policy(st_t).max(1)[1].item()
            s, r, d, info = env.step(a)
            st += 1
        if info.get("reason") == "goal":
            eval_succ += 1
    print(f"Eval SR: {eval_succ}/100 = {eval_succ}%")

    # Save
    torch.save(policy.state_dict(), "dqn_dwa_v2_500.pth")
    print("Model saved to dqn_dwa_v2_500.pth")

if __name__ == "__main__":
    train()

