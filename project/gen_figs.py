"""gen_figs.py — Regenerate all paper figures (图1–图5), including 三合一 (E1AC).

Usage: python gen_figs.py

Figures:
  图1 fig_paths_20x20_all.png   - 20x20 path comparison (7 methods incl. E1AC)
  图2 fig_reward_20x20_all.png  - 20x20 reward convergence (7 methods incl. E1AC)
  图3 fig_paths_30x30_all.png   - 30x30 path comparison (7 methods incl. E1AC)
  图4 fig_trad_compare_{20,30}.png - traditional comparison (RRT / A* / E1A)
  图5 fig_noise_robustness.png  - noise robustness (MLP vs CAT)

Notes:
  - DDQN/Dueling/PER models & logs live in ROOT; all other files live in project/.
  - Dueling uses a value/advantage architecture (needs its own loader).
  - ASGS methods (E1A, E1AC) apply the final Q-penalty (lambda=15) in greedy eval.
"""
import sys, os, math, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import torch
import torch.nn as nn

ROOT = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main"
PROJ = os.path.join(ROOT, "project")
sys.path.insert(0, PROJ)

from models import CrossAttentionDQN
from astar_rrt_compare import astar, rrt

device = torch.device("cpu")
SEQ_LEN = 5
ASGS_LAM = 15.0  # final lambda for ASGS greedy eval (matches paper methodology)

# CLI: python gen_figs.py [dpi] [outdir]
DPI = int(sys.argv[1]) if len(sys.argv) > 1 else 150
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else PROJ
os.makedirs(OUTDIR, exist_ok=True)


# ---------------- Networks / loaders ----------------
class MLPDQN(nn.Module):
    """MLP with fc1/fc2/fc3 keys (used by train_compare.py for DDQN/PER)."""
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
    """Dueling network with value/advantage streams (train_compare.py)."""
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


def load_mlp(fname):
    """Load MLP saved either as nn.Sequential (train_large) or fc1/fc2/fc3 (train_compare)."""
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


def load_dueling(fname):
    net = DuelingDQN().to(device)
    net.load_state_dict(torch.load(fname, map_location=device, weights_only=True))
    net.eval()
    return net


def load_cat(fname):
    net = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)
    net.load_state_dict(torch.load(fname, map_location=device, weights_only=True))
    net.eval()
    return net


def load_net(fname, kind):
    if kind == "mlp":
        return load_mlp(fname)
    if kind == "dueling":
        return load_dueling(fname)
    if kind == "cat":
        return load_cat(fname)
    raise ValueError(kind)


def resolve(fname, loc):
    base = ROOT if loc == "root" else PROJ
    return os.path.join(base, fname)


# ---------------- Greedy rollout ----------------
def run_path(net, env, use_cat=False, use_asgs=False):
    s = env.reset()
    hist = deque([s] * SEQ_LEN, maxlen=SEQ_LEN) if use_cat else None
    path = [(env.agent_pos[0], env.agent_pos[1])]
    done, st = False, 0
    while not done and st < 500:
        with torch.no_grad():
            if use_cat:
                seq = torch.FloatTensor(np.stack(list(hist))).unsqueeze(0).to(device)
                q = net(seq)
            else:
                q = net(torch.FloatTensor(s).unsqueeze(0).to(device))
            if use_asgs:
                q = q - torch.FloatTensor(s[3:11]).unsqueeze(0).to(device) * ASGS_LAM
            a = q.max(1)[1].item()
        ns, _, done, info = env.step(a)
        if use_cat:
            hist.append(ns)
        s = ns
        st += 1
        path.append((env.agent_pos[0], env.agent_pos[1]))
    return path, info


# ---------------- Drawing ----------------
def draw_grid(ax, env):
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    for i in range(env.size + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.4)
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.4)
    for (ox, oy) in env.obstacles:
        ax.add_patch(Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor='black', edgecolor='none'))
    ax.plot(env.start[0], env.start[1], 'o', color='blue', markersize=9, zorder=5)
    ax.plot(env.goal[0], env.goal[1], 's', color='green', markersize=9, zorder=5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_paths(ax, env, path, title, color='red'):
    draw_grid(ax, env)
    if path:
        arr = np.array(path)
        ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2, alpha=0.9)
    ax.set_title(title, fontsize=10)


# ---------------- Method configs ----------------
# (label, model filename, kind, use_asgs, location)
methods20 = [
    ("S0 Sparse",        "dqn_S0_sparse_baseline_20x20_10000.pth",  "mlp",     False, "project"),
    ("E1 Coupled",       "dqn_E1_coupled_20x20_10000.pth",          "mlp",     False, "project"),
    ("E1A Coupled+ASGS", "dqn_E1_coupled_ASGS_20x20_10000.pth",     "mlp",     True,  "project"),
    ("DDQN",             "dqn_DDQN_20x20_coupled_10000.pth",        "mlp",     False, "root"),
    ("Dueling DQN",      "dqn_DUELING_20x20_coupled_10000.pth",     "dueling", False, "root"),
    ("PER DQN",          "dqn_PER_20x20_coupled_10000.pth",         "mlp",     False, "root"),
    ("E1AC CAT+CRF+ASGS","dqn_E1_CAT_ASGS_20x20_10000.pth",         "cat",     True,  "project"),
]

methods30 = [
    ("S0 Sparse",        "dqn_S0_sparse_baseline_30x30_10000.pth",  "mlp",     False, "project"),
    ("E1 Coupled",       "dqn_E1_coupled_30x30_10000.pth",          "mlp",     False, "project"),
    ("E1A Coupled+ASGS", "dqn_E1_coupled_ASGS_30x30_10000.pth",     "mlp",     True,  "project"),
    ("DDQN",             "dqn_DDQN_30x30_coupled_10000.pth",        "mlp",     False, "root"),
    ("Dueling DQN",      "dqn_DUELING_30x30_coupled_10000.pth",     "dueling", False, "root"),
    ("PER DQN",          "dqn_PER_30x30_coupled_10000.pth",         "mlp",     False, "root"),
    ("E1AC CAT+CRF+ASGS","dqn_E1_CAT_ASGS_30x30_10000.pth",         "cat",     True,  "project"),
]

logs20 = [
    ("S0 Sparse",        "S0_sparse_baseline_20x20_reward_log.csv",  "project", "tab:red"),
    ("E1 Coupled",       "E1_coupled_20x20_reward_log.csv",          "project", "tab:blue"),
    ("E1A Coupled+ASGS", "E1_coupled_ASGS_20x20_reward_log.csv",     "project", "tab:green"),
    ("DDQN",             "DDQN_20x20_coupled_reward_log.csv",        "root",    "tab:purple"),
    ("Dueling DQN",      "DUELING_20x20_coupled_reward_log.csv",     "root",    "tab:orange"),
    ("PER DQN",          "PER_20x20_coupled_reward_log.csv",         "root",    "tab:brown"),
    ("E1AC CAT+CRF+ASGS","E1_CAT_ASGS_20x20_reward_log.csv",         "project", "tab:cyan"),
]


def make_path_figure(methods, env, out_name, title_suffix):
    n = len(methods)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.atleast_1d(axes).flatten()
    for ax, (label, mfile, kind, use_asgs, loc) in zip(axes, methods):
        try:
            net = load_net(resolve(mfile, loc), kind)
            path, info = run_path(net, env, use_cat=(kind == "cat"), use_asgs=use_asgs)
            plot_paths(ax, env, path, f"{label} ({len(path) - 1} steps)")
        except Exception as e:
            ax.text(0.5, 0.5, f"{label}: missing", ha='center', va='center')
            ax.set_title(label, fontsize=10)
    for ax in axes[len(methods):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, out_name), dpi=DPI)
    plt.close()
    print(f"Saved {out_name}")


def make_reward_figure(logs, out_name):
    def moving_avg(d, w=100):
        if len(d) < w:
            return d
        return np.convolve(d, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, logf, loc, color in logs:
        p = resolve(logf, loc)
        if os.path.exists(p):
            d = np.loadtxt(p)
            m = moving_avg(d)
            ax.plot(np.arange(len(m)), m, color=color, label=label, linewidth=1.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (moving avg)")
    ax.set_title("20x20 Reward Convergence (all methods)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, out_name), dpi=DPI)
    plt.close()
    print(f"Saved {out_name}")


def make_traditional_figures():
    from env_large import GridEnvLarge
    from env_large30 import GridEnv30
    random.seed(42)
    configs = [
        ("20x20", GridEnvLarge(), "dqn_E1_coupled_ASGS_20x20_10000.pth"),
        ("30x30", GridEnv30(), "dqn_E1_coupled_ASGS_30x30_10000.pth"),
    ]
    for name, env, mfile in configs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_paths(axes[0], env, rrt(env), "RRT", "purple")
        plot_paths(axes[1], env, astar(env), "A*", "blue")
        net = load_mlp(os.path.join(PROJ, mfile))
        path, _ = run_path(net, env, use_asgs=True)
        plot_paths(axes[2], env, path, "E1A (ours)", "red")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"fig_trad_compare_{name}.png"), dpi=DPI)
        plt.close()
        print(f"Saved fig_trad_compare_{name}.png")


def make_noise_figure():
    import json
    json_path = os.path.join(PROJ, "noise_robustness_results.json")
    if not os.path.exists(json_path):
        print("SKIP fig_noise_robustness.png: run eval_noise_multiseed.py first "
              "(missing noise_robustness_results.json)")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    noise_levels = res["noise_levels"]

    def mean_std(key):
        means = [np.mean(res[key][str(lv)]) for lv in noise_levels]
        stds = [np.std(res[key][str(lv)]) for lv in noise_levels]
        return means, stds

    mlp_mean, mlp_std = mean_std("mlp")
    cat_mean, cat_std = mean_std("cat")

    x = [lv * 100 for lv in noise_levels]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, mlp_mean, yerr=mlp_std, fmt='o-', color='tab:red',
                label='MLP', linewidth=1.5, capsize=3)
    ax.errorbar(x, cat_mean, yerr=cat_std, fmt='s-', color='tab:blue',
                label='CAT', linewidth=1.5, capsize=3)
    ax.set_xlabel("Sensor noise level (%)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Noise robustness (clean-trained models, mean±std)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_noise_robustness.png"), dpi=DPI)
    plt.close()
    print("Saved fig_noise_robustness.png")


if __name__ == "__main__":
    from env_large import GridEnvLarge
    from env_large30 import GridEnv30

    make_path_figure(methods20, GridEnvLarge(), "fig_paths_20x20_all.png", "20x20")
    make_path_figure(methods30, GridEnv30(), "fig_paths_30x30_all.png", "30x30")
    make_reward_figure(logs20, "fig_reward_20x20_all.png")
    make_traditional_figures()
    make_noise_figure()
    print("ALL FIGURES DONE")
