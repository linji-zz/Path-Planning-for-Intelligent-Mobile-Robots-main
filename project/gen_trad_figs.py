"""gen_trad_figs.py - Generate A*/RRT/E1A path comparison figures."""
import sys, os, math, random, heapq, torch, torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")
from astar_rrt_compare import astar, rrt

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
    return path

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

def plot_path(ax, env, path, title, color):
    draw_grid(ax, env)
    if path:
        arr = np.array(path)
        ax.plot(arr[:,0], arr[:,1], color=color, linewidth=2, alpha=0.9)
    ax.set_title(title, fontsize=10)

def main():
    from env_large import GridEnvLarge
    from env_large30 import GridEnv30

    random.seed(42)
    configs = [
        ("20x20", GridEnvLarge(), "dqn_E1_coupled_ASGS_20x20_10000.pth"),
        ("30x30", GridEnv30(), "dqn_E1_coupled_ASGS_30x30_10000.pth"),
    ]

    for name, env, mfile in configs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # RRT
        rrt_path = rrt(env)
        plot_path(axes[0], env, rrt_path, f"RRT", "purple")
        # A*
        astar_path = astar(env)
        plot_path(axes[1], env, astar_path, f"A*", "blue")
        # E1A
        net = load_mlp(os.path.join(BASE, mfile))
        e1a_path = run_path(net, env)
        plot_path(axes[2], env, e1a_path, f"E1A (ours)", "red")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, f"fig_trad_compare_{name}.png"), dpi=150)
        plt.close()
        print(f"Saved fig_trad_compare_{name}.png")

if __name__ == "__main__":
    main()
