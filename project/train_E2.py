import sys, numpy as np, random, math, time, os
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")
from env1 import GridEnv
from models import CrossAttentionDQN

device = torch.device("cpu")
print("E2: CAT + 耦合奖励 | 15x15 | 4000 eps", flush=True)

env = GridEnv(); SEQ_LEN = 5
policy = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)
target = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)
target.load_state_dict(policy.state_dict()); target.eval()
opt = optim.Adam(policy.parameters(), lr=0.001)
mem = deque(maxlen=20000)

def openness(env,tx,ty):
    cnt=0
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            nx,ny=tx+dx,ty+dy
            if 0<=nx<env.size and 0<=ny<env.size and (nx,ny) in env.obstacles: cnt+=1
    return 1.0-cnt/9.0

EPISODES, MAX_STEPS, BATCH = 4000, 300, 128
GAMMA, EPS_START, EPS_END, EPS_DECAY = 0.9, 0.9, 0.05, 0.997
TAU = 0.005
eps, start_t = EPS_START, time.time()
successes, rewards = [], []
base = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project"

for ep in range(EPISODES):
    s = env.reset(); h = deque([s]*SEQ_LEN, maxlen=SEQ_LEN)
    tr, done, steps, ag_theta = 0, False, 0, 0.0
    while not done and steps < MAX_STEPS:
        if random.random() < eps:
            a = random.randrange(8)
        else:
            with torch.no_grad():
                seq = torch.FloatTensor(np.stack(list(h))).unsqueeze(0).to(device)
                a = policy(seq).max(1)[1].item()
        old_pos = env.agent_pos; dx, dy = env.actions[a]
        sub_goal = (old_pos[0]+dx, old_pos[1]+dy)
        ns, r_orig, done, info = env.step(a)
        old_d = math.hypot(old_pos[0]-env.goal[0], old_pos[1]-env.goal[1])
        new_d = math.hypot(env.agent_pos[0]-env.goal[0], env.agent_pos[1]-env.goal[1])
        r = (old_d-new_d)*2.0
        ang = math.atan2(sub_goal[1]-env.agent_pos[1], sub_goal[0]-env.agent_pos[0])
        r += 0.5*math.cos(ag_theta-ang) + 0.3*openness(env,int(round(sub_goal[0])),int(round(sub_goal[1])))
        if done:
            if info.get("reason")=="goal": r=50.0
            elif info.get("reason") in ("collision","boundary"): r=-10.0
        if new_d<old_d:
            ag_theta = math.atan2(env.agent_pos[1]-old_pos[1], env.agent_pos[0]-old_pos[0])
        tr+=r
        nh = list(h)[1:] + [ns]
        mem.append((np.stack(list(h)).astype(np.float32), a, np.array(nh,dtype=np.float32), r, done))
        h.append(ns); s = ns; steps+=1
        if len(mem)>=BATCH:
            batch = list(zip(*random.sample(mem,BATCH)))
            sb=torch.FloatTensor(np.array(batch[0])).to(device)
            ab=torch.LongTensor(batch[1]).unsqueeze(1).to(device)
            nsb=torch.FloatTensor(np.array(batch[2])).to(device)
            rb=torch.FloatTensor(batch[3]).unsqueeze(1).to(device)
            db=torch.FloatTensor(batch[4]).unsqueeze(1).to(device)
            curr=policy(sb).gather(1,ab)
            with torch.no_grad():
                tgt=rb+(1-db)*GAMMA*target(nsb).max(1,keepdim=True)[0]
            nn.MSELoss()(curr,tgt).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        for tp,p in zip(target.parameters(),policy.parameters()):
            tp.data.copy_(TAU*p.data+(1-TAU)*tp.data)
    if eps>EPS_END: eps*=EPS_DECAY
    rewards.append(tr); successes.append(info.get("reason")=="goal")
    
    # Save checkpoint every 500 episodes
    if (ep+1)%500==0:
        avg_r=np.mean(rewards[-500:]); sr=sum(successes[-500:])/5
        torch.save(policy.state_dict(), os.path.join(base, f"dqn_E2_ckpt_{ep+1}.pth"))
        print(f"E2 Ep {ep+1}/{EPISODES} | AvgRew: {avg_r:.1f} | SR: {sr:.1f}% | Eps: {eps:.3f} | CKPT saved | Time: {time.time()-start_t:.0f}s", flush=True)

t=time.time()-start_t; print(f"E2 Done! Time: {t:.0f}s", flush=True)
print("E2 Eval (200 eps)...", flush=True)
policy.eval(); sg, lg = 0, []
for ep in range(200):
    s=env.reset(); h=deque([s]*SEQ_LEN, maxlen=SEQ_LEN); done, st=False, 0
    while not done and st<MAX_STEPS:
        with torch.no_grad():
            seq=torch.FloatTensor(np.stack(list(h))).unsqueeze(0).to(device)
            a=policy(seq).max(1)[1].item()
        ns,_,done,info=env.step(a); h.append(ns); s=ns; st+=1
    if info.get("reason")=="goal": sg+=1; lg.append(st)
avg_len=np.mean(lg) if lg else 0
print(f"E2 Greedy: {sg}/200 = {sg/2:.1f}% | AvgLen: {avg_len:.1f}", flush=True)
torch.save(policy.state_dict(), os.path.join(base, "dqn_E2_4000.pth"))
print("E2 Final model saved.", flush=True)
