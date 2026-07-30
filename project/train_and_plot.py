import sys, numpy as np, random, math, time, csv
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
sys.path.insert(0, '.')
from env1 import GridEnv

device = torch.device('cpu')
env = GridEnv()

def openness(env, tx, ty):
    cnt = 0
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            nx,ny=tx+dx,ty+dy
            if 0<=nx<env.size and 0<=ny<env.size and (nx,ny) in env.obstacles: cnt+=1
    return 1.0-cnt/9.0

def train_and_log(name, use_coupled):
    print(f'Training {name}...', flush=True)
    policy = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8)).to(device)
    target = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8)).to(device)
    target.load_state_dict(policy.state_dict()); target.eval()
    opt = optim.Adam(policy.parameters(), lr=0.001)
    mem = deque(maxlen=20000)
    
    EPISODES, MAX_STEPS, BATCH = 5000, 300, 128
    GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU = 0.9, 0.9, 0.05, 0.997, 0.005
    eps = EPS_START
    start_t = time.time()
    reward_log = []
    
    for ep in range(EPISODES):
        s = env.reset()
        tr, done, steps, ag_theta = 0, False, 0, 0.0
        while not done and steps < MAX_STEPS:
            a = random.randrange(8) if random.random()<eps else policy(torch.FloatTensor(s).unsqueeze(0)).max(1)[1].item()
            old_pos = env.agent_pos; dx, dy = env.actions[a]
            sub = (old_pos[0]+dx, old_pos[1]+dy)
            ns, r_orig, done, info = env.step(a)
            
            if use_coupled:
                old_d = math.hypot(old_pos[0]-env.goal[0], old_pos[1]-env.goal[1])
                new_d = math.hypot(env.agent_pos[0]-env.goal[0], env.agent_pos[1]-env.goal[1])
                r = (old_d-new_d)*2.0
                ang = math.atan2(sub[1]-env.agent_pos[1], sub[0]-env.agent_pos[0])
                r += 0.5*math.cos(ag_theta-ang)
                r += 0.3*openness(env, int(round(sub[0])), int(round(sub[1])))
                if done:
                    if info.get('reason')=='goal': r=50.0
                    elif info.get('reason') in ('collision','boundary'): r=-10.0
                if new_d<old_d:
                    ag_theta = math.atan2(env.agent_pos[1]-old_pos[1], env.agent_pos[0]-old_pos[0])
            else:
                r = r_orig
            
            tr += r
            mem.append((s,a,ns,r,done))
            s = ns; steps += 1
            if len(mem) >= BATCH:
                batch = list(zip(*random.sample(mem, BATCH)))
                sb = torch.FloatTensor(np.array(batch[0]))
                ab = torch.LongTensor(batch[1]).unsqueeze(1)
                nsb = torch.FloatTensor(np.array(batch[2]))
                rb = torch.FloatTensor(batch[3]).unsqueeze(1)
                db = torch.FloatTensor(batch[4]).unsqueeze(1)
                curr = policy(sb).gather(1,ab)
                with torch.no_grad(): tgt = rb+(1-db)*GAMMA*target(nsb).max(1,keepdim=True)[0]
                nn.MSELoss()(curr,tgt).backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            for tp,p in zip(target.parameters(),policy.parameters()):
                tp.data.copy_(TAU*p.data+(1-TAU)*tp.data)
        if eps > EPS_END: eps *= EPS_DECAY
        
        reward_log.append(tr)
        if (ep+1) % 500 == 0:
            avg = np.mean(reward_log[-500:])
            print(f'  {name} Ep {ep+1}/{EPISODES} | AvgRew: {avg:.1f} | Eps: {eps:.3f} | Time: {time.time()-start_t:.0f}s', flush=True)
    
    # Save reward log
    np.savetxt(f'{name}_reward_log.csv', np.array(reward_log), delimiter=',')
    print(f'  {name} saved. Total time: {time.time()-start_t:.0f}s', flush=True)
    return reward_log

# Train E0 and E1
train_and_log('E0_original', False)
train_and_log('E1_coupled', True)

# Now plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    e0 = np.loadtxt('E0_original_reward_log.csv', delimiter=',')
    e1 = np.loadtxt('E1_coupled_reward_log.csv', delimiter=',')
    
    window = 50
    e0_smooth = np.convolve(e0, np.ones(window)/window, mode='valid')
    e1_smooth = np.convolve(e1, np.ones(window)/window, mode='valid')
    eps_range = np.arange(window-1, 5000)
    
    plt.figure(figsize=(10, 6))
    plt.plot(eps_range, e0_smooth, 'b-', linewidth=2, label='E0: MLP + Original Reward')
    plt.plot(eps_range, e1_smooth, 'r-', linewidth=2, label='E1: MLP + Coupled Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (window=50)')
    plt.title('Training Reward Comparison - 15x15 Grid')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_comparison.png', dpi=150)
    print('Plot saved: reward_comparison.png', flush=True)
except Exception as e:
    print(f'Plot error (non-fatal): {e}', flush=True)

print('Done!', flush=True)
