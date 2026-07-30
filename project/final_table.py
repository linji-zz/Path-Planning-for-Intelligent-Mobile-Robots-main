import sys, numpy as np, torch, torch.nn as nn
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

def load_net(fname):
    sd = torch.load(fname, map_location="cpu", weights_only=True)
    if "fc1.weight" in sd:
        net = DQN().to(device); net.load_state_dict(sd); return net, False
    elif "ag_proj.weight" in sd:
        net = CrossAttentionDQN().to(device); net.load_state_dict(sd); return net, True
    else:
        mapping = {"0":"fc1","2":"fc2","4":"fc3"}
        sd2 = {}
        for k, v in sd.items():
            parts = k.split(".")
            new_k = mapping.get(parts[0], parts[0]) + "." + parts[1]
            sd2[new_k] = v
        net = DQN().to(device); net.load_state_dict(sd2); return net, False

def evaluate(fname):
    net, is_cat = load_net(fname)
    net.eval()
    sg, lg = 0, []
    for ep in range(200):
        s = env.reset()
        h = deque([s]*5, maxlen=5) if is_cat else None
        done, st = False, 0
        while not done and st < 300:
            with torch.no_grad():
                if is_cat:
                    seq = torch.FloatTensor(np.stack(list(h))).unsqueeze(0)
                    a = net(seq).max(1)[1].item()
                else:
                    a = net(torch.FloatTensor(s).unsqueeze(0)).max(1)[1].item()
            ns, _, done, info = env.step(a)
            if is_cat: h.append(ns)
            s = ns; st += 1
        if info.get("reason") == "goal": sg += 1; lg.append(st)
    return sg/2, np.mean(lg) if lg else 0

EXPS = [
    ("dqn_env1_baseline_10000.pth", "E0: MLP + Original Reward", "10000"),
    ("dqn_coupled_10000.pth",       "E1: MLP + Coupled Reward", "10000"),
    ("dqn_E2_5500.pth",             "E2: CAT + Coupled Reward", "5500"),
    ("dqn_A1_no_openness_10000.pth","A1: Coupled - Openness",   "10000"),
    ("dqn_A2_no_alignment_10000.pth","A2: Coupled - Alignment", "10000"),
]

print("=" * 80)
print("15x15 Grid | Final Comparison | 200 greedy test episodes")
print("=" * 80)
print("{:<30} {:<20} {:<15} {:<10}".format("Method", "Training Episodes", "Success Rate", "Path Len"))
print("-" * 80)

for fname, name, eps in EXPS:
    sr, l = evaluate(fname)
    name_str = name.ljust(30)
    eps_str = eps.ljust(18)
    sr_str = f"{sr:.1f}%".ljust(13)
    print(f"{name_str} {eps_str} {sr_str} {l:.1f}")

print("=" * 80)
