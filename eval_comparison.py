import numpy as np
import math
import torch
import torch.nn as nn
import time
import sys, os

sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

from env1 import GridEnv as DiscreteEnv
from dwa_env import DWAEnv

class DQN(nn.Module):
    def __init__(self, state_dim=11, action_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

configs = [
    {
        "name": "原始 DQN (离散)",
        "model_file": "dqn_env1_baseline_10000.pth",
        "env_type": "discrete",
        "coupled_reward": False,
        "dwa_params": None,
    },
    {
        "name": "耦合奖励 DQN (离散)",
        "model_file": "dqn_coupled_10000.pth",
        "env_type": "discrete",
        "coupled_reward": False,
        "dwa_params": None,
    },
    {
        "name": "DQN + DWA (纯距离奖励)",
        "model_file": "dqn_dwa_10000.pth",
        "env_type": "dwa",
        "coupled_reward": False,
        "dwa_params": {"v_max": 2.0, "w_max": 2.0, "v_acc": 1.0, "w_acc": 1.0, "dt": 0.1, "eval_time": 0.5},
    },
    {
        "name": "DQN + DWA + 耦合奖励",
        "model_file": "dqn_coupled_dwa_opt_10000.pth",
        "env_type": "dwa",
        "coupled_reward": True,
        "dwa_params": {"v_max": 2.0, "w_max": 2.0, "v_acc": 1.0, "w_acc": 1.0, "dt": 0.1, "eval_time": 0.5},
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main"
model_dir = os.path.join(base_dir, "project")
NUM_TEST = 200
MAX_STEPS = 300
results = {}

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Evaluating: {cfg['name']}")
    print(f"{'='*60}")

    policy_net = DQN().to(device)
    model_path = os.path.join(model_dir, cfg["model_file"])
    if not os.path.exists(model_path):
        print(f"  [SKIP] model not found: {model_path}")
        continue
    sd = torch.load(model_path, map_location=device, weights_only=True)
    policy_net.load_state_dict(sd)
    policy_net.eval()

    successes, rewards, lengths, dists = 0, [], [], []
    collisions, boundaries = 0, 0
    success_lengths = []

    for ep in range(NUM_TEST):
        if cfg["env_type"] == "discrete":
            env = DiscreteEnv()
        else:
            env = DWAEnv(dwa_params=cfg["dwa_params"], coupled_reward=cfg["coupled_reward"])

        state = env.reset()
        done = False
        steps = 0
        total_r = 0

        while not done and steps < MAX_STEPS:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(device)
                a = policy_net(st).max(1)[1].item()
            ns, r, done, info = env.step(a)
            total_r += r
            state = ns
            steps += 1

        rewards.append(total_r)
        lengths.append(steps)
        d = math.hypot(env.agent_pos[0] - env.goal[0], env.agent_pos[1] - env.goal[1])
        dists.append(d)

        if info.get("reason") == "goal":
            successes += 1
            success_lengths.append(steps)
        elif info.get("reason") == "collision":
            collisions += 1
        elif info.get("reason") == "boundary":
            boundaries += 1

    sr = successes / NUM_TEST * 100
    avg_r = np.mean(rewards)
    avg_l = np.mean(lengths)
    avg_d = np.mean(dists)
    avg_sl = np.mean(success_lengths) if success_lengths else 0

    print(f"  Success Rate:  {successes}/{NUM_TEST} = {sr:.1f}%")
    print(f"  Avg Reward:    {avg_r:.2f}")
    print(f"  Avg Path:      {avg_l:.2f}")
    print(f"  Success Path:  {avg_sl:.2f}")
    print(f"  Avg Distance:  {avg_d:.2f}")
    print(f"  Collisions:    {collisions}")
    print(f"  Boundaries:    {boundaries}")

    results[cfg["name"]] = {
        "sr": sr, "avg_r": avg_r, "avg_l": avg_l,
        "avg_sl": avg_sl, "avg_d": avg_d,
        "collisions": collisions, "boundaries": boundaries,
    }

print(f"\n\n{'='*70}")
print(f"COMPARISON SUMMARY ({NUM_TEST} episodes each)")
print(f"{'='*70}")
print(f"{'Config':<30} {'Success%':<10} {'AvgRew':<10} {'PathLen':<10} {'SuccLen':<10} {'Collision':<10}")
print(f"{'-'*80}")
for n, r in results.items():
    print(f"{n:<30} {r['sr']:<8.1f}%  {r['avg_r']:<10.2f} {r['avg_l']:<10.2f} {r['avg_sl']:<10.2f} {r['collisions']:<10}")
