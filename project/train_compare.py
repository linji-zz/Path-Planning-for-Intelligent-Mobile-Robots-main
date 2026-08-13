"""train_compare.py - Baseline DQN variants for comparison.
Usage: python train_compare.py <method> <map> <seed>
  method: ddqn | dueling | per
  map: 20 | 30
All methods use the coupled reward (same as E1) for fair comparison.
"""
import sys, numpy as np, random, math, time, os
from collections import deque
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

METHOD = sys.argv[1] if len(sys.argv) > 1 else "ddqn"
MAPSIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1

if METHOD not in ("ddqn", "dueling", "per"):
    print("Usage: python train_compare.py ddqn|dueling|per <20|30> <seed>")
    sys.exit(1)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

if MAPSIZE == 20:
    from env_large import GridEnvLarge as Env
    env = Env(seed=SEED)
    MAX_STEPS = 400
else:
    from env_large30 import GridEnv30 as Env
    env = Env()
    MAX_STEPS = 500

NAME = f"{METHOD.upper()}_{MAPSIZE}x{MAPSIZE}_coupled"
if SEED > 1:
    NAME = f"{NAME}_run{SEED}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60, flush=True)
print(f"Experiment: {NAME} | Method: {METHOD} | Map: {MAPSIZE}x{MAPSIZE} | Seed: {SEED}", flush=True)
print("="*60, flush=True)

# ---------------- Networks ----------------
class MLPDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.v = nn.Linear(128, 1)
        self.a = nn.Linear(128, 8)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.v(x)
        adv = self.a(x)
        return v + (adv - adv.mean(dim=1, keepdim=True))

if METHOD == "dueling":
    def make_net():
        return DuelingDQN().to(device)
else:
    def make_net():
        return MLPDQN().to(device)

policy = make_net()
target = make_net()
target.load_state_dict(policy.state_dict())
target.eval()
opt = optim.Adam(policy.parameters(), lr=0.001)

# ---------------- PER Memory (SumTree) ----------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = [None]*capacity
        self.n_entries = 0
        self.write = 0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left = 2*idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    def total(self):
        return self.tree[0]
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

class PERMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_inc=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.eps = 1e-6
    def push(self, td_error, transition):
        p = (abs(td_error) + self.eps) ** self.alpha
        self.tree.add(p, transition)
    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
        self.beta = min(1.0, self.beta + self.beta_inc)
        # importance sampling weights
        weights = []
        total = self.tree.total()
        for idx in idxs:
            p = self.tree.tree[idx]
            w = (self.capacity * p / total) ** (-self.beta)
            weights.append(w)
        w_max = max(weights) if weights else 1.0
        weights = [w / w_max for w in weights]
        return batch, idxs, weights
    def update_priorities(self, idxs, td_errors):
        for idx, td in zip(idxs, td_errors):
            p = (abs(td) + self.eps) ** self.alpha
            self.tree.update(idx, p)

if METHOD == "per":
    memory = PERMemory(30000)
else:
    memory = deque(maxlen=30000)

# ---------------- Helpers ----------------
def openness(env, tx, ty):
    cnt = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = tx+dx, ty+dy
            if 0 <= nx < env.size and 0 <= ny < env.size and (nx, ny) in env.obstacles:
                cnt += 1
    return 1.0 - cnt/9.0

EPISODES, BATCH = 10000, 128
GAMMA, EPS_START, EPS_END, EPS_DECAY = 0.9, 0.9, 0.05, 0.998
TAU = 0.005
eps, start_t = EPS_START, time.time()
successes, rewards, reward_log = [], [], []

for ep in range(EPISODES):
    s = env.reset()
    tr, done, steps, ag_theta = 0, False, 0, 0.0
    while not done and steps < MAX_STEPS:
        if random.random() < eps:
            action = random.randrange(8)
        else:
            with torch.no_grad():
                q = policy(torch.FloatTensor(s).unsqueeze(0).to(device))
            action = q.max(1)[1].item()
        old_pos = env.agent_pos
        dx, dy = env.actions[action]
        sub_goal = (old_pos[0]+dx, old_pos[1]+dy)
        ns, r_orig, done, info = env.step(action)
        old_d = math.hypot(old_pos[0]-env.goal[0], old_pos[1]-env.goal[1])
        new_d = math.hypot(env.agent_pos[0]-env.goal[0], env.agent_pos[1]-env.goal[1])
        # coupled reward (same as E1)
        r_dist = (old_d-new_d)*2.0
        ang = math.atan2(sub_goal[1]-env.agent_pos[1], sub_goal[0]-env.agent_pos[0])
        r_align = 0.5 * math.cos(ag_theta - ang)
        opn = openness(env, int(round(sub_goal[0])), int(round(sub_goal[1])))
        r_open = 0.3 * opn
        reward = r_dist + r_align + r_open
        if done:
            if info.get("reason")=="goal": reward = 50.0
            elif info.get("reason") in ("collision","boundary"): reward = -10.0
        if new_d < old_d:
            ag_theta = math.atan2(env.agent_pos[1]-old_pos[1], env.agent_pos[0]-old_pos[0])
        tr += reward
        if METHOD == "per":
            with torch.no_grad():
                q_cur = policy(torch.FloatTensor(s).unsqueeze(0).to(device))[0, action].item()
                q_next = target(torch.FloatTensor(ns).unsqueeze(0).to(device)).max(1)[0].item()
                td = reward + (1-int(done))*GAMMA*q_next - q_cur
            memory.push(td, (s.astype(np.float32), action, ns.astype(np.float32), reward, done))
        else:
            memory.append((s.astype(np.float32), action, ns.astype(np.float32), reward, done))
        s = ns; steps += 1

        if (METHOD == "per" and memory.tree.n_entries >= BATCH) or (METHOD != "per" and len(memory) >= BATCH):
            if METHOD == "per":
                batch, idxs, weights = memory.sample(BATCH)
            else:
                batch = random.sample(memory, BATCH)
                weights = [1.0]*BATCH
                idxs = None
            sb = torch.FloatTensor(np.array([m[0] for m in batch])).to(device)
            ab = torch.LongTensor([m[1] for m in batch]).unsqueeze(1).to(device)
            nsb = torch.FloatTensor(np.array([m[2] for m in batch])).to(device)
            rb = torch.FloatTensor([m[3] for m in batch]).unsqueeze(1).to(device)
            db = torch.FloatTensor([m[4] for m in batch]).unsqueeze(1).to(device)
            wb = torch.FloatTensor(weights).unsqueeze(1).to(device)
            curr = policy(sb).gather(1, ab)
            with torch.no_grad():
                if METHOD == "ddqn":
                    # Double DQN: policy selects, target evaluates
                    next_a = policy(nsb).max(1, keepdim=True)[1]
                    tgt = rb + (1-db)*GAMMA*target(nsb).gather(1, next_a)
                else:
                    # DQN / Dueling / PER: standard max
                    tgt = rb + (1-db)*GAMMA*target(nsb).max(1, keepdim=True)[0]
            loss = (wb * (curr - tgt).pow(2)).mean()
            loss.backward(); opt.step(); opt.zero_grad()
            if METHOD == "per":
                with torch.no_grad():
                    td_errs = (curr - tgt).detach().abs().squeeze(1).cpu().numpy()
                memory.update_priorities(idxs, td_errs)

    with torch.no_grad():
        for tp, p in zip(target.parameters(), policy.parameters()):
            tp.data.copy_(TAU*p.data+(1-TAU)*tp.data)
    if eps > EPS_END: eps *= EPS_DECAY
    rewards.append(tr)
    successes.append(info.get("reason")=="goal")
    reward_log.append(tr)
    if (ep+1)%200==0:
        avg_r = np.mean(rewards[-200:])
        sr = sum(successes[-200:])/2
        elapsed = time.time()-start_t
        print(f"{NAME} Ep {ep+1}/{EPISODES} | AvgRew: {avg_r:.1f} | SR: {sr:.1f}% | Eps: {eps:.3f} | Elapsed: {elapsed:.0f}s", flush=True)

t = time.time()-start_t; print(f"{NAME} Done! Time: {t:.0f}s", flush=True)
np.savetxt(f"{NAME}_reward_log.csv", np.array(reward_log), fmt="%.18e")
print(f"Reward log saved to {NAME}_reward_log.csv", flush=True)

# Eval
print(f"{NAME} Eval (greedy, 200 eps)...", flush=True)
policy.eval(); sg, lg = 0, []
for ep in range(200):
    s = env.reset(); done, st = False, 0
    while not done and st < MAX_STEPS:
        with torch.no_grad():
            a = policy(torch.FloatTensor(s).unsqueeze(0).to(device)).max(1)[1].item()
        ns, _, done, info = env.step(a)
        s = ns; st += 1
    if info.get("reason")=="goal": sg+=1; lg.append(st)
avg_len = np.mean(lg) if lg else 0
print(f"{NAME} Greedy: {sg}/200 = {sg/2:.1f}% | AvgLen: {avg_len:.1f}", flush=True)
torch.save(policy.state_dict(), f"dqn_{NAME}_10000.pth")
print(f"{NAME} Model saved.", flush=True)
