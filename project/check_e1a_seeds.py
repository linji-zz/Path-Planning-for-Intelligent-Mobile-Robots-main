"""check_e1a_seeds.py - Check path steps and turns for E1A across 3 seeds."""
import sys, torch, torch.nn as nn
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
        path.append((int(round(env.agent_pos[0])), int(round(env.agent_pos[1]))))
        s = ns; st += 1
    return path, info

def path_stats(path):
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

    envs20 = [GridEnvLarge() for _ in range(3)]
    envs30 = [GridEnv30() for _ in range(3)]

    print("="*70)
    print("E1A path quality across 3 seeds (greedy eval)")
    print("="*70)

    print("--- 20x20 (A*: 22 steps, 4 turns) ---")
    best20 = None
    for seed in [None, 2, 3]:
        if seed is None:
            mfile = "dqn_E1_coupled_ASGS_20x20_10000.pth"
            tag = "seed1"
        else:
            mfile = f"dqn_E1_coupled_ASGS_20x20_run{seed}_10000.pth"
            tag = f"seed{seed}"
        net = load_mlp(BASE + "\\" + mfile)
        path, info = run_path(net, envs20[0])
        steps, turns = path_stats(path)
        print(f"  {tag}: {steps} steps, {turns} turns ({info.get('reason')})")
        if best20 is None or turns < best20[1]:
            best20 = (tag, turns, steps)

    print("--- 30x30 (A*: 33 steps, 12 turns) ---")
    best30 = None
    for seed in [None, 2, 3]:
        if seed is None:
            mfile = "dqn_E1_coupled_ASGS_30x30_10000.pth"
            tag = "seed1"
        else:
            mfile = f"dqn_E1_coupled_ASGS_30x30_run{seed}_10000.pth"
            tag = f"seed{seed}"
        net = load_mlp(BASE + "\\" + mfile)
        path, info = run_path(net, envs30[0])
        steps, turns = path_stats(path)
        print(f"  {tag}: {steps} steps, {turns} turns ({info.get('reason')})")
        if best30 is None or turns < best30[1]:
            best30 = (tag, turns, steps)

    print("="*70)
    print(f"Best 20x20: {best20[0]} ({best20[2]} steps, {best20[1]} turns)")
    print(f"Best 30x30: {best30[0]} ({best30[2]} steps, {best30[1]} turns)")

if __name__ == "__main__":
    main()
