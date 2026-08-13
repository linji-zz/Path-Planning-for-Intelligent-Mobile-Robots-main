"""eval_traditional_compare.py - Compute path steps and turns for E1A vs A* vs RRT.
Loads trained E1A models, runs greedy path, computes steps and turn counts.
"""
import sys, math, torch, torch.nn as nn
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
    """Run greedy policy, return path of integer cells."""
    s = env.reset()
    path = [(env.agent_pos[0], env.agent_pos[1])]
    done, st = False, 0
    while not done and st < 500:
        with torch.no_grad():
            a = net(torch.FloatTensor(s).unsqueeze(0)).max(1)[1].item()
        ns, _, done, info = env.step(a)
        path.append((int(round(env.agent_pos[0])), int(round(env.agent_pos[1]))))
        s = ns; st += 1
    return path, info

def path_stats(path):
    """Compute steps and turn count on integer grid path."""
    if path is None or len(path) < 2:
        return 0, 0
    steps = len(path) - 1
    turns = 0
    prev_dir = None
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        cur_dir = (dx, dy)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return steps, turns

def main():
    from env_large import GridEnvLarge
    from env_large30 import GridEnv30

    configs = [
        ("20x20", GridEnvLarge(), "dqn_E1_coupled_ASGS_20x20_10000.pth"),
        ("30x30", GridEnv30(), "dqn_E1_coupled_ASGS_30x30_10000.pth"),
    ]

    # A* and RRT results from astar_rrt_compare.py
    trad = {
        "20x20": {"A*": (22, 4), "RRT": (74.4, "-")},
        "30x30": {"A*": (33, 12), "RRT": (103.5, "-")},
    }

    print("="*70)
    print("TRADITIONAL ALGORITHM COMPARISON (path steps / turns)")
    print("="*70)
    for name, env, mfile in configs:
        net = load_mlf(BASE + "\\" + mfile) if False else load_mlp(BASE + "\\" + mfile)
        path, info = run_path(net, env)
        steps, turns = path_stats(path)
        a_steps, a_turns = trad[name]["A*"]
        r_steps, r_turns = trad[name]["RRT"]
        print(f"--- {name} (E1A reason: {info.get('reason')}) ---")
        print(f"{'Algorithm':<10} {'Steps':<12} {'Turns':<10}")
        print(f"{'RRT':<10} {str(r_steps):<12} {str(r_turns):<10}")
        print(f"{'A*':<10} {str(a_steps):<12} {str(a_turns):<10}")
        print(f"{'E1A (ours)':<10} {str(steps):<12} {str(turns):<10}")
        # improvements
        r_imp = (1 - steps/r_steps)*100
        a_imp = (1 - steps/a_steps)*100
        a_turn_diff = a_turns - turns
        print(f"E1A vs RRT: steps {r_imp:.1f}% shorter")
        print(f"E1A vs A* : steps {a_imp:.1f}% (0=equal), turns {a_turn_diff} fewer")
        print()

if __name__ == "__main__":
    main()
