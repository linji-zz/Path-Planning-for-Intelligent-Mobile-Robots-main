import sys, numpy as np, random, math, time
from collections import deque
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")
from dwa_env import DWAEnv

DWA_PARAMS = {"v_max":2.0,"w_max":2.0,"v_acc":1.0,"w_acc":1.0,"dt":0.1,"eval_time":0.5}

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8))
    def forward(self,x): return self.net(x)

class ReplayMemory:
    def __init__(self,cap): self.memory=deque(maxlen=cap)
    def push(self,t): self.memory.append(t)
    def sample(self,bs): return random.sample(self.memory,bs)
    def __len__(self): return len(self.memory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Reduce DWA planner sample_num for speed
from dwa_planner import DWAPlanner
DWAPlanner.sample_num = 8  # Override

env = DWAEnv(dwa_params=DWA_PARAMS, coupled_reward=False)
policy = DQN().to(device)
target = DQN().to(device)
target.load_state_dict(policy.state_dict())
target.eval()
opt = optim.Adam(policy.parameters(), lr=0.001)
mem = ReplayMemory(20000)

EPISODES = 1000
MAX_STEPS = 200
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START, EPS_END, EPS_DECAY = 0.9, 0.05, 0.998
TAU = 0.005
eps = EPS_START
train_sr, rewards_hist = [], []
start_t = time.time()

for ep in range(EPISODES):
    s = env.reset()
    tr, done, steps = 0, False, 0
    while not done and steps < MAX_STEPS:
        if random.random() < eps:
            a = random.randrange(8)
        else:
            with torch.no_grad():
                a = policy(torch.FloatTensor(s).unsqueeze(0).to(device)).max(1)[1].item()
        ns, r, done, info = env.step(a)
        tr += r
        mem.push((s,a,ns,r,done))
        s = ns
        steps += 1
        if len(mem) >= BATCH_SIZE:
            batch = list(zip(*mem.sample(BATCH_SIZE)))
            sb=torch.FloatTensor(np.array(batch[0])).to(device)
            ab=torch.LongTensor(batch[1]).unsqueeze(1).to(device)
            nsb=torch.FloatTensor(np.array(batch[2])).to(device)
            rb=torch.FloatTensor(batch[3]).unsqueeze(1).to(device)
            db=torch.FloatTensor(batch[4]).unsqueeze(1).to(device)
            curr=policy(sb).gather(1,ab)
            with torch.no_grad():
                tgt=rb+(1-db)*GAMMA*target(nsb).max(1,keepdim=True)[0]
            nn.MSELoss()(curr,tgt).backward()
            opt.step()
            opt.zero_grad()
    with torch.no_grad():
        for tp,p in zip(target.parameters(),policy.parameters()):
            tp.data.copy_(TAU*p.data+(1-TAU)*tp.data)
    if eps > EPS_END: eps *= EPS_DECAY
    rewards_hist.append(tr)
    train_sr.append(info.get("reason")=="goal")
    if (ep+1)%200==0:
        sr=sum(train_sr[-200:])/2
        avg_r=np.mean(rewards_hist[-200:])
        print(f"Ep {ep+1}/{EPISODES} | AvgRew: {avg_r:.1f} | SR: {sr:.1f}% | Eps: {eps:.3f} | Time: {time.time()-start_t:.0f}s", flush=True)

t=time.time()-start_t
print(f"\nDone! Time: {t:.0f}s", flush=True)
print(f"Training SR (overall): {sum(train_sr)/EPISODES*100:.1f}%", flush=True)

# Greedy eval
print("\n===== Greedy Eval =====", flush=True)
policy.eval()
sr_g, lens_g = 0, []
for ep in range(200):
    s=env.reset()
    done, steps=False, 0
    while not done and steps<MAX_STEPS:
        a=policy(torch.FloatTensor(s).unsqueeze(0).to(device)).max(1)[1].item()
        s,_,done,info=env.step(a)
        steps+=1
    if info.get("reason")=="goal":
        sr_g+=1
        lens_g.append(steps)
avg_len=np.mean(lens_g) if lens_g else 0
print(f"Greedy: {sr_g}/200 = {sr_g/2:.1f}% | Avg success len: {avg_len:.1f}", flush=True)
torch.save(policy.state_dict(), "dqn_dwa_v2_pure_1000.pth")
print("Model saved.", flush=True)
