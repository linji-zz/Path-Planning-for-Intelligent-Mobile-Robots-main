import sys, math, numpy as np, torch, torch.nn as nn
from collections import deque
sys.path.insert(0, r'C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project')
from env1 import GridEnv
from models import CrossAttentionDQN

device = torch.device('cpu')
env = GridEnv(); MAX_STEPS = 300

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

def eval_model(model_file, is_cat=False):
    if is_cat:
        net = CrossAttentionDQN().to(device)
    else:
        net = DQN().to(device)
    sd = torch.load(model_file, map_location=device, weights_only=True)
    net.load_state_dict(sd); net.eval()
    sg, lg, rew_g = 0, [], []
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
        rew_g.append(tr)
        if info.get('reason') == 'goal': sg += 1; lg.append(st)
    return sg/2, np.mean(lg) if lg else 0, np.mean(rew_g)

models = [
    ('dqn_env1_baseline_10000.pth', 'E0: MLP + Original Reward', False),
    ('dqn_coupled_10000.pth', 'E1: MLP + Coupled Reward', False),
    ('dqn_E2_best.pth', 'E2: CAT + Coupled (3000ep)', True),
    ('dqn_A1_no_openness.pth', 'A1: Coupled - Openness', False),
    ('dqn_A2_no_alignment.pth', 'A2: Coupled - Alignment', False),
]

print('=' * 85)
print('15x15 Grid Experiment Results (200 test episodes, greedy policy)')
print('=' * 85)
print(f"{'Experiment':<35} {'Success Rate':<15} {'Path Length':<13} {'Avg Reward'}")
print('-' * 85)

for fname, label, is_cat in models:
    sr, l, r = eval_model(fname, is_cat)
    print(f'{label:<35} {sr:<13.1f}%  {l:<13.1f} {r:<.1f}')

print('=' * 85)
