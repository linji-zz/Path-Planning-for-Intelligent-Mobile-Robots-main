"""gen_paper_figs2.py - Generate path figures for all algorithms, traditional comparison, reward curves."""
import sys, os, math, torch, torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

BASE = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project"
device = torch.device("cpu")

def load_mlp(fname):
    sd = torch.load(fname, map_location=device, weights_only=True)
    if list(sd.keys())[0].startswith("fc"):
        m = {"fc1.weight":"0.weight","fc1.bias":"0.bias","fc2.weight":"2.weight","fc2.bias":"2.bias","fc3.weight":"4.weight","fc3.bias":"4.bias"}
        sd = {m[k]:v for k,v in sd.items()}
    net = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8)).to(device)
    net.load_state_dict(sd); net.eval()
    return net

def run_path(net, env):
    s = env.reset()
    path = [(env.agent_pos[0], env.agent_pos[1])]
    done, st = False, 0
    while not done and st < 500:
        with torch.no_grad():
            a = net(torch.FloatTensor(s).unsqueeze(0)).max(1)[1].item()
        ns, _, done, info = env.step(a)
        path.append((env.agent_pos[0], env.agent_pos[1]))
        s = ns; st += 1
    return path, info

def draw_grid(ax, env):
    ax.set_xlim(-0.5, env.size-0.5); ax.set_ylim(-0.5, env.size-0.5)
    for i in range(env.size+1):
        ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.4)
        ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.4)
    for (ox, oy) in env.obstacles:
        ax.add_patch(Rectangle((ox-0.5, oy-0.5), 1, 1, facecolor='black', edgecolor='none'))
    ax.plot(env.start[0], env.start[1], 'o', color='blue', markersize=9, zorder=5)
    ax.plot(env.goal[0], env.goal[1], 's', color='green', markersize=9, zorder=5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

def plot_paths(ax, env, path, title, color='red'):
    draw_grid(ax, env)
    if path:
        arr = np.array(path)
        ax.plot(arr[:,0], arr[:,1], color=color, linewidth=2, alpha=0.9)
    ax.set_title(title, fontsize=10)

# Config: name, model suffix
exps20 = [
    ("S0 Sparse", "dqn_S0_sparse_baseline_20x20_10000.pth"),
    ("E1 Coupled", "dqn_E1_coupled_20x20_10000.pth"),
    ("E1A Coupled+ASGS", "dqn_E1_coupled_ASGS_20x20_10000.pth"),
    ("DDQN", "dqn_DDQN_20x20_coupled_10000.pth"),
    ("Dueling", "dqn_DUELING_20x20_coupled_10000.pth"),
    ("PER", "dqn_PER_20x20_coupled_10000.pth"),
]
exps30 = [
    ("S0 Sparse", "dqn_S0_sparse_baseline_30x30_10000.pth"),
    ("E1 Coupled", "dqn_E1_coupled_30x30_10000.pth"),
    ("E1A Coupled+ASGS", "dqn_E1_coupled_ASGS_30x30_10000.pth"),
    ("DDQN", "dqn_DDQN_30x30_coupled_10000.pth"),
    ("Dueling", "dqn_DUELING_30x30_coupled_10000.pth"),
    ("PER", "dqn_PER_30x30_coupled_10000.pth"),
]

from env_large import GridEnvLarge
from env_large30 import GridEnv30

# Figure 1: 20x20 all algorithm paths
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
env = GridEnvLarge()
for ax, (name, mfile) in zip(axes.flatten(), exps20):
    try:
        net = load_mlp(os.path.join(BASE, mfile))
        path, info = run_path(net, env)
        plot_paths(ax, env, path, f"{name} ({len(path)-1} steps)")
    except Exception as e:
        ax.text(0.5, 0.5, f"{name}: missing", ha='center')
        ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "fig_paths_20x20_all.png"), dpi=150)
plt.close()
print("Saved fig_paths_20x20_all.png")

# Figure 2: 30x30 all algorithm paths
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
env = GridEnv30()
for ax, (name, mfile) in zip(axes.flatten(), exps30):
    try:
        net = load_mlp(os.path.join(BASE, mfile))
        path, info = run_path(net, env)
        plot_paths(ax, env, path, f"{name} ({len(path)-1} steps)")
    except Exception as e:
        ax.text(0.5, 0.5, f"{name}: missing", ha='center')
        ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "fig_paths_30x30_all.png"), dpi=150)
plt.close()
print("Saved fig_paths_30x30_all.png")

# Figure 3: reward convergence curves with comparison methods
fig, ax = plt.subplots(figsize=(10, 6))
def moving_avg(d, w=100):
    if len(d) < w: return d
    return np.convolve(d, np.ones(w)/w, mode="valid")

def load_log20(mfile):
    name = os.path.splitext(os.path.basename(mfile))[0]
    # derive log name
    return None

# Log files for reward curves (run1 only)
logs20 = [
    ("S0 Sparse", "S0_sparse_baseline_20x20_reward_log.csv", "tab:red"),
    ("E1 Coupled", "E1_coupled_20x20_reward_log.csv", "tab:blue"),
    ("E1A Coupled+ASGS", "E1_coupled_ASGS_20x20_reward_log.csv", "tab:green"),
    ("DDQN", "DDQN_20x20_coupled_reward_log.csv", "tab:purple"),
    ("Dueling", "DUELING_20x20_coupled_reward_log.csv", "tab:orange"),
    ("PER", "PER_20x20_coupled_reward_log.csv", "tab:brown"),
]
for name, logf, color in logs20:
    p = os.path.join(BASE, logf)
    if os.path.exists(p):
        d = np.loadtxt(p)
        m = moving_avg(d)
        ax.plot(np.arange(len(m)), m, color=color, label=name, linewidth=1.3)
ax.set_xlabel("Episode"); ax.set_ylabel("Reward (moving avg)")
ax.set_title("20x20 Reward Convergence (all methods)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "fig_reward_20x20_all.png"), dpi=150)
plt.close()
print("Saved fig_reward_20x20_all.png")

print("ALL FIGURES DONE")
