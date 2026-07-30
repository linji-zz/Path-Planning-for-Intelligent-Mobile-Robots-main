import sys, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")
from dwa_env import DWAEnv
from env1 import GridEnv as DiscreteEnv

DWA_PARAMS = {"v_max":2.0,"w_max":2.0,"v_acc":1.0,"w_acc":1.0,"dt":0.1,"eval_time":0.5}

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project"

# Load DWA model (saved with "net.0.weight" prefix - strip it)
sd_dwa_raw = torch.load(f"{base}/dqn_dwa_v2_pure_1000.pth", map_location=device, weights_only=True)
sd_dwa = {".".join(k.split(".")[1:]): v for k, v in sd_dwa_raw.items()}  # strip "net." prefix

net_dwa = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8))
net_dwa.load_state_dict(sd_dwa)
net_dwa.eval().to(device)

# Load discrete model
net_disc = DQN().to(device)
sd_disc = torch.load(f"{base}/dqn_env1_baseline_10000.pth", map_location=device, weights_only=True)
net_disc.load_state_dict(sd_disc)
net_disc.eval()

MAX_STEPS = 300
print(f"{'='*70}")
print(f"{'配置':<32} {'成功率':<10} {'路径步数':<12} {'行进距离':<10}")
print(f"{'-'*70}")

# 1. Original DQN (discrete)
succ, steps_l, dist_l = 0, [], []
for ep in range(200):
    env = DiscreteEnv()
    s = env.reset()
    done, st = False, 0; plen = 0.0
    while not done and st < MAX_STEPS:
        a = net_disc(torch.FloatTensor(s).unsqueeze(0).to(device)).max(1)[1].item()
        old = env.agent_pos; s, r, done, info = env.step(a)
        if old != env.agent_pos: plen += math.hypot(env.agent_pos[0]-old[0], env.agent_pos[1]-old[1])
        st += 1
    if info.get("reason")=="goal": succ += 1; steps_l.append(st); dist_l.append(plen)
avg_st = np.mean(steps_l) if steps_l else 0; avg_d = np.mean(dist_l) if dist_l else 0
print(f"{'DQN (离散, 原始距离)':<32} {succ/2:<8.1f}%  {avg_st:<12.1f} {avg_d:<10.1f}")

# 2. DQN + DWA v2 (pure distance, fixed)
succ, steps_l, dist_l = 0, [], []
for ep in range(200):
    env = DWAEnv(dwa_params=DWA_PARAMS, coupled_reward=False)
    s = env.reset(); done, st = False, 0; plen = 0.0
    while not done and st < MAX_STEPS:
        a = net_dwa(torch.FloatTensor(s).unsqueeze(0).to(device)).max(1)[1].item()
        ox, oy = env.cont_x, env.cont_y
        s, r, done, info = env.step(a)
        plen += math.hypot(env.cont_x-ox, env.cont_y-oy)
        st += 1
    if info.get("reason")=="goal": succ += 1; steps_l.append(st); dist_l.append(plen)
avg_st = np.mean(steps_l) if steps_l else 0; avg_d = np.mean(dist_l) if dist_l else 0
print(f"{'DQN + DWA v2 (纯距离, 修复)':<32} {succ/2:<8.1f}%  {avg_st:<12.1f} {avg_d:<10.1f}")

# 3. Coupled reward + DWA (the original version for comparison)
net_coupled = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8))
try:
    sd_coup = torch.load(f"{base}/dqn_coupled_dwa_opt_10000.pth", map_location=device, weights_only=True)
    sd_coup = {".".join(k.split(".")[1:]): v for k, v in sd_coup.items()}
    net_coupled.load_state_dict(sd_coup)
    net_coupled.eval().to(device)
    succ, steps_l, dist_l = 0, [], []
    for ep in range(200):
        env = DWAEnv(dwa_params=DWA_PARAMS, coupled_reward=True)
        s = env.reset(); done, st = False, 0; plen = 0.0
        while not done and st < MAX_STEPS:
            a = net_coupled(torch.FloatTensor(s).unsqueeze(0).to(device)).max(1)[1].item()
            ox, oy = env.cont_x, env.cont_y
            s, r, done, info = env.step(a)
            plen += math.hypot(env.cont_x-ox, env.cont_y-oy)
            st += 1
        if info.get("reason")=="goal": succ += 1; steps_l.append(st); dist_l.append(plen)
    avg_st = np.mean(steps_l) if steps_l else 0; avg_d = np.mean(dist_l) if dist_l else 0
    print(f"{'DQN + DWA (旧版 耦合奖励)':<32} {succ/2:<8.1f}%  {avg_st:<12.1f} {avg_d:<10.1f}")
except:
    print(f"{'DQN + DWA (旧版 耦合奖励)':<32} {'N/A (旧DWA环境不兼容)':<30}")

print(f"{'-'*70}")
print("v2修复: heading改进, current_theta传入, 完整sub_steps, 障碍物安全钳位, 全速采样, 全点碰撞检测")
print(f"{'='*70}")
