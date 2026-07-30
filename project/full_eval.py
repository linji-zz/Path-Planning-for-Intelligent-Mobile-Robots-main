import numpy as np, math, sys, torch, torch.nn as nn
from collections import deque
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")
from env_large import GridEnvLarge
from models import CrossAttentionDQN

BASE = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project"
device = torch.device("cpu")
env = GridEnvLarge()
MAX_STEPS = 400

# Model definitions
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)
    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.fc3(x)

def load_model(fname, is_cat=False):
    sd = torch.load(fname, map_location=device, weights_only=True)
    if is_cat:
        net = CrossAttentionDQN().to(device)
        net.load_state_dict(sd)
    elif list(sd.keys())[0].startswith("fc"):
        net = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8)).to(device)
        m = {"fc1.weight":"0.weight","fc1.bias":"0.bias","fc2.weight":"2.weight","fc2.bias":"2.bias","fc3.weight":"4.weight","fc3.bias":"4.bias"}
        sd2 = {m[k]:v for k,v in sd.items()}
        net.load_state_dict(sd2)
    else:
        net = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8)).to(device)
        net.load_state_dict(sd)
    net.eval(); return net

def evaluate(fname, is_cat=False):
    net = load_model(fname, is_cat)
    sg, lg, coll, rew_l = 0, [], 0, []
    for ep in range(200):
        s = env.reset()
        h = deque([s]*5, maxlen=5) if is_cat else None
        done, st, tr = False, 0, 0.0
        while not done and st < MAX_STEPS:
            with torch.no_grad():
                if is_cat:
                    seq = torch.FloatTensor(np.stack(list(h))).unsqueeze(0)
                    a = net(seq).max(1)[1].item()
                else:
                    a = net(torch.FloatTensor(s).unsqueeze(0)).max(1)[1].item()
            ns, r, done, info = env.step(a)
            if is_cat: h.append(ns)
            tr += r; s = ns; st += 1
        rew_l.append(tr)
        if info.get("reason")=="goal": sg += 1; lg.append(st)
        elif info.get("reason") in ("collision","boundary"): coll += 1
    sr = sg/2
    avg_len = round(np.mean(lg),1) if lg else 0
    avg_rew = round(np.mean(rew_l),1)
    return sr, avg_len, avg_rew, coll

def compute_convergence(log_file, window=100, threshold_ratio=0.95):
    d = np.loadtxt(log_file)
    total = len(d)
    best = max(np.mean(d[i:i+50]) for i in range(len(d)-50)) if len(d)>=50 else np.mean(d)
    target = threshold_ratio * best
    conv = total
    for i in range(window, total-window):
        if np.mean(d[i:i+window]) >= target:
            conv = i + window
            break
    return conv, round(best,1), total

experiments = [
    ("E0: Original", BASE+"/dqn_E0_original_20x20_10000.pth", BASE+"/E0_original_20x20_reward_log.csv", False, 356),
    ("E1: Coupled",  BASE+"/dqn_E1_coupled_20x20_10000.pth", BASE+"/E1_coupled_20x20_reward_log.csv", False, 354),
    ("A1: NoOpen",   BASE+"/dqn_A1_no_openness_20x20_10000.pth", BASE+"/A1_no_openness_20x20_reward_log.csv", False, 363),
    ("A2: NoAlign",  BASE+"/dqn_A2_no_alignment_20x20_10000.pth", BASE+"/A2_no_alignment_20x20_reward_log.csv", False, 366),
    ("E2: CAT", None, None, True, None),
]

import os

print("="*110)
print("COMPREHENSIVE EXPERIMENT RESULTS | 20x20 Grid | 10000 episodes")
print("="*110)
print("{:<18} {:<10} {:<12} {:<10} {:<12} {:<12} {:<12} {:<10}".format(
    "Experiment","Time(s)","Conv(Ep)","Best50","Success","PathLen","AvgRew","Collide"))
print("-"*110)

for name, modelf, logf, is_cat, t in experiments:
    if modelf is None or not os.path.exists(modelf):
        print(f"{'E2: CAT':<18} {'N/A (running)':<10} {'-':<12} {'-':<10} {'-':<12} {'-':<12} {'-':<12} {'-':<10}")
        continue
    sr, avg_len, avg_rew, coll = evaluate(modelf, is_cat)
    if logf and os.path.exists(logf):
        conv, best50, total = compute_convergence(logf)
    else:
        conv, best50 = "-", "-"
    t_str = str(t) if t else "-"
    print(f"{name:<18} {t_str:<10} {str(conv):<12} {str(best50):<10} {sr:<10.1f}% {avg_len:<12.1f} {avg_rew:<12.1f} {coll:<10}")

print("="*110)
print()
print("METRIC DEFINITIONS:")
print("  Time(s)    : Wall-clock training time (seconds)")
print("  Conv(Ep)   : Episode to reach 95% of best 50-ep reward (convergence speed)")
print("  Best50     : Best 50-episode sliding window reward")
print("  Success    : Greedy test success rate (200 episodes)")
print("  PathLen    : Average successful path steps")
print("  AvgRew     : Average episode reward (greedy test)")
print("  Collide    : Number of collisions in 200 test episodes")
