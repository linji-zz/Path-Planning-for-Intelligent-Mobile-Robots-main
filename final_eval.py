import sys, math, numpy as np, torch, torch.nn as nn
from collections import deque
sys.path.insert(0, ".")
from env1 import GridEnv
from models import CrossAttentionDQN

device = torch.device("cpu")
env = GridEnv()

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def load_model(fname):
    sd = torch.load(fname, map_location="cpu", weights_only=True)
    if "fc1.weight" in sd:
        net = DQN().to(device); net.load_state_dict(sd); return net, False
    elif "0.weight" in sd:
        sd2 = {}
        mapping = {"0": "fc1", "2": "fc2", "4": "fc3"}
        for k, v in sd.items():
            parts = k.split(".")
            new_key = mapping.get(parts[0], parts[0]) + "." + parts[1]
            sd2[new_key] = v
        net = DQN().to(device); net.load_state_dict(sd2); return net, False
    else:
        net = CrossAttentionDQN().to(device); net.load_state_dict(sd); return net, True

def evaluate(fname):
    net, is_cat = load_model(fname)
    net.eval()
    sg, lg, rw = 0, [], []
    for ep in range(200):
        s = env.reset()
        h = deque([s]*5, maxlen=5) if is_cat else None
        done, st, tr = False, 0, 0.0
        while not done and st < 300:
            with torch.no_grad():
                if is_cat:
                    seq = torch.FloatTensor(np.stack(list(h))).unsqueeze(0)
                    a = net(seq).max(1)[1].item()
                else:
                    a = net(torch.FloatTensor(s).unsqueeze(0)).max(1)[1].item()
            ns, r, done, info = env.step(a)
            if is_cat: h.append(ns)
            tr += r; s = ns; st += 1
        rw.append(tr)
        if info.get("reason") == "goal": sg += 1; lg.append(st)
    return sg/2, np.mean(lg) if lg else 0, np.mean(rw)

MODELS = [
    ("dqn_env1_baseline_10000.pth", "E0: MLP + Original Reward"),
    ("dqn_coupled_10000.pth",     "E1: MLP + Coupled Reward"),
    ("dqn_E2_best.pth",           "E2: CAT + Coupled (3000ep)"),
    ("dqn_A1_no_openness.pth",    "A1: Coupled Less Openness"),
    ("dqn_A2_no_alignment.pth",   "A2: Coupled Less Alignment"),
]

print("=" * 85)
print("Final Results | 15x15 | 200 greedy episodes")
print("=" * 85)
print("{:<28} {:<15} {:<12} {}".format("Experiment", "Success Rate", "Path Length", "Avg Reward"))
print("-" * 85)
for f, name in MODELS:
    sr, l, r = evaluate(f)
    print("{:<28} {:<13.1f}%  {:<12.1f} {:.1f}".format(name, sr, l, r))
print("=" * 85)
