"""eval_noise_30x30.py — 30x30 noise robustness: E1 (MLP) vs E2 (CAT), both NO ASGS.

Exact 30x30 analogue of eval_noise_multiseed.py (which ran E1 vs E2 on 20x20).
Both models use coupled reward, no ASGS, so no Q-penalty in greedy eval.
Isolates CAT's noise-robustness contribution on the 30x30 map.
Results saved to noise_robustness_results_30x30_noasgs.json.
"""
import sys, os, json, random
import numpy as np
import torch, torch.nn as nn
from collections import deque

ROOT = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main"
PROJ = os.path.join(ROOT, "project")
sys.path.insert(0, PROJ)
sys.path.insert(0, ROOT)

from models import CrossAttentionDQN
from env_large30 import GridEnv30

device = torch.device("cpu")
SEQ_LEN = 5
MAX_STEPS = 500


def load_mlp(fname):
    sd = torch.load(fname, map_location=device, weights_only=True)
    net = nn.Sequential(nn.Linear(11, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 8)).to(device)
    if list(sd.keys())[0].startswith("fc"):
        m = {"fc1.weight": "0.weight", "fc1.bias": "0.bias",
             "fc2.weight": "2.weight", "fc2.bias": "2.bias",
             "fc3.weight": "4.weight", "fc3.bias": "4.bias"}
        sd = {m[k]: v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net


def load_cat(fname):
    net = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)
    net.load_state_dict(torch.load(fname, map_location=device, weights_only=True))
    net.eval()
    return net


def eval_noise(net, use_cat, level, episodes, seed):
    env = GridEnv30()
    env.set_noise(level)
    random.seed(seed)
    sg = 0
    for _ in range(episodes):
        s = env.reset()
        hist = deque([s] * SEQ_LEN, maxlen=SEQ_LEN) if use_cat else None
        done, st = False, 0
        while not done and st < MAX_STEPS:
            with torch.no_grad():
                if use_cat:
                    seq = torch.FloatTensor(np.stack(list(hist))).unsqueeze(0).to(device)
                    q = net(seq)
                else:
                    q = net(torch.FloatTensor(s).unsqueeze(0).to(device))
                a = q.max(1)[1].item()
            ns, _, done, info = env.step(a)
            if use_cat:
                hist.append(ns)
            s = ns
            st += 1
        if info.get("reason") == "goal":
            sg += 1
    return sg / episodes * 100.0


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seeds = list(range(n_seeds))
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    mlp = load_mlp(os.path.join(PROJ, "dqn_E1_coupled_30x30_10000.pth"))
    cat = load_cat(os.path.join(PROJ, "dqn_E2_CAT_coupled_30x30_10000.pth"))

    results = {"noise_levels": noise_levels, "seeds": seeds,
               "episodes": episodes, "mlp": {}, "cat": {}}
    print(f"30x30 noise eval (E1 MLP vs E2 CAT, no ASGS) | seeds={seeds} | eps/seed={episodes}")
    print("-" * 60)
    for lv in noise_levels:
        mlp_vals, cat_vals = [], []
        for sd in seeds:
            mlp_vals.append(eval_noise(mlp, False, lv, episodes, sd))
            cat_vals.append(eval_noise(cat, True, lv, episodes, sd))
        results["mlp"][str(lv)] = mlp_vals
        results["cat"][str(lv)] = cat_vals
        m_mean, m_std = np.mean(mlp_vals), np.std(mlp_vals)
        c_mean, c_std = np.mean(cat_vals), np.std(cat_vals)
        print(f"noise={lv*100:>2.0f}%  E1(MLP)={m_mean:5.1f}±{m_std:4.1f}  E2(CAT)={c_mean:5.1f}±{c_std:4.1f}")

    out = os.path.join(PROJ, "noise_robustness_results_30x30_noasgs.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
