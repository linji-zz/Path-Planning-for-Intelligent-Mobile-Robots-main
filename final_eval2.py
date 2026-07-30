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

def load_model(fname, is_cat=False):
    sd = torch.load(fname, map_location="cpu", weights_only=True)
    if is_cat:
        net = CrossAttentionDQN().to(device)
    elif "fc1.weight" in sd:
        net = DQN().to(device)
    else:
        m = {"0":"fc1","2":"fc2","4":"fc3"}
        sd = {m[k.split(".")[0]] + "." + k.split(".")[1]: v for k, v in sd.items()}
        net = DQN().to(device)
    net.load_state_dict(sd)
    net.eval()
    return net, is_cat

def evaluate(fname, is_cat=False):
    net, cat = load_model(fname, is_cat)
    sg, lg = 0, []
    for ep in range(200):
        s = env.reset()
        h = deque([s]*5, maxlen=5) if cat else None
        done, st = False, 0
        while not done and st < 300:
            with torch.no_grad():
                if cat:
                    seq = torch.FloatTensor(np.stack(list(h))).unsqueeze(0)
                    a = net(seq).max(1)[1].item()
                else:
                    a = net(torch.FloatTensor(s).unsqueeze(0)).max(1)[1].item()
            ns, _, done, info = env.step(a)
            if cat: h.append(ns)
            s = ns; st += 1
        if info.get("reason") == "goal": sg += 1; lg.append(st)
    return sg/2, np.mean(lg) if lg else 0

EXPS = [
    ("dqn_env1_baseline_10000.pth",  "E0: MLP + Original Distance",  "10000", False),
    ("dqn_coupled_10000.pth",        "E1: MLP + Coupled Reward",    "10000", False),
    ("dqn_E2_5500.pth",              "E2: CAT + Coupled Reward",    "5500",  True),
    ("dqn_A1_no_openness_10000.pth", "A1: Coupled - Openness",      "10000", False),
    ("dqn_A2_no_alignment_10000.pth","A2: Coupled - Alignment",     "10000", False),
]

print("=" * 85)
print("FINAL COMPARISON | 15x15 Grid | 200 greedy episodes")
print("=" * 85)
h = "{:<30} {:<8} {:<10} {:<10} {}".format("Model", "Eps", "Success", "Steps", "Train Reward")
print(h)
print("-" * 85)
for fname, name, eps, cat in EXPS:
    sr, l = evaluate(fname, cat)
    print("{:<30} {:<8} {:<10.1f} {:<10.1f}".format(name, eps, sr, l))
print("=" * 85)
