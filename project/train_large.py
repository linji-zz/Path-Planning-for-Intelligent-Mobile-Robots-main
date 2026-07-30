import sys, numpy as np, random, math, time, os

from collections import deque

import torch, torch.nn as nn, torch.optim as optim



sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

from env_large import GridEnvLarge

from models import CrossAttentionDQN



EXPS = {

    "S0": {"coupled": False, "cat": False, "name": "S0_sparse_baseline_20x20", "sparse": True},
    "S0A": {"coupled": False, "cat": False, "name": "S0_ASGS_20x20", "sparse": True, "asgs": True},

    "E1": {"coupled": True,  "cat": False, "name": "E1_coupled_20x20"},
    "E1A": {"coupled": True,  "cat": False, "name": "E1_coupled_ASGS_20x20", "asgs": True},

    "A1": {"coupled": "no_openness", "cat": False, "name": "A1_no_openness_20x20"},

    "A2": {"coupled": "no_alignment", "cat": False, "name": "A2_no_alignment_20x20"},

    "E2": {"coupled": True,  "cat": True,  "name": "E2_CAT_coupled_20x20"},

}



if len(sys.argv) < 2 or sys.argv[1] not in EXPS:

    print("Usage: python train_large.py S0|S0A|E1|E1A|A1|A2|E2")

    sys.exit(1)



cfg = EXPS[sys.argv[1]]

NAME = cfg["name"]

USE_COUPLED = cfg["coupled"]

USE_CAT = cfg["cat"]
SPARSE = cfg.get("sparse", False)
ASGS = cfg.get("asgs", False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60, flush=True)

print(f"Experiment: {NAME} | Coupled={USE_COUPLED} | CAT={USE_CAT} | Device={device}", flush=True)

print("="*60, flush=True)



env = GridEnvLarge()
if SPARSE:
    env.set_sparse(True)

SEQ_LEN = 5 if USE_CAT else 1

state_dim, action_dim = 11, 8



if USE_CAT:

    policy = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)

else:

    policy = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8)).to(device)

if USE_CAT:

    target = CrossAttentionDQN(d_model=64, nhead=4, seq_len=SEQ_LEN).to(device)

else:

    target = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8)).to(device)

target.load_state_dict(policy.state_dict())

target.eval()

opt = optim.Adam(policy.parameters(), lr=0.001)

memory = deque(maxlen=30000)



def openness(env, tx, ty):

    cnt=0

    for dx in (-1,0,1):

        for dy in (-1,0,1):

            nx,ny=tx+dx,ty+dy

            if 0<=nx<env.size and 0<=ny<env.size and (nx,ny) in env.obstacles: cnt+=1

    return 1.0-cnt/9.0



EPISODES, MAX_STEPS, BATCH = 10000, 400, 128

GAMMA, EPS_START, EPS_END, EPS_DECAY = 0.9, 0.9, 0.05, 0.998

TAU = 0.005

eps, start_t = EPS_START, time.time()

successes, rewards, reward_log = [], [], []



for ep in range(EPISODES):

    s = env.reset()

    history = deque([s]*SEQ_LEN, maxlen=SEQ_LEN)

    tr, done, steps, ag_theta = 0, False, 0, 0.0



    while not done and steps < MAX_STEPS:

        obs = s[3:11]
        if random.random() < eps:


            if ASGS:
                weights_list = [0.1 if o > 0.5 else 1.0 for o in obs]
                total_w = sum(weights_list)
                r = random.random() * total_w
                cum = 0
                action = 0
                for iw, w in enumerate(weights_list):
                    cum += w
                    if r <= cum:
                        action = iw
                        break
            else:
                action = random.randrange(action_dim)
        else:

            with torch.no_grad():

                if USE_CAT:

                    seq = torch.FloatTensor(np.stack(list(history))).unsqueeze(0).to(device)

                    q = policy(seq)

                    
                    if ASGS:
                        lam = 5.0 + 10.0 * ep / EPISODES
                        q = q - torch.FloatTensor(obs).unsqueeze(0).to(device) * lam
                    
                    action = q.max(1)[1].item()
                else:

                    st = torch.FloatTensor(s).unsqueeze(0).to(device)

                    q = policy(st)



                    
                    if ASGS:
                        lam = 5.0 + 10.0 * ep / EPISODES
                        q = q - torch.FloatTensor(obs).unsqueeze(0).to(device) * lam
                    
                    action = q.max(1)[1].item()
        old_pos = env.agent_pos

        dx, dy = env.actions[action]

        sub_goal = (old_pos[0]+dx, old_pos[1]+dy)

        ns, r_orig, done, info = env.step(action)



        old_d = math.hypot(old_pos[0]-env.goal[0], old_pos[1]-env.goal[1])
    
        new_d = math.hypot(env.agent_pos[0]-env.goal[0], env.agent_pos[1]-env.goal[1])
    
        if USE_COUPLED:
            r_dist = (old_d-new_d)*2.0
    


            if USE_COUPLED == "no_openness":

                ang = math.atan2(sub_goal[1]-env.agent_pos[1], sub_goal[0]-env.agent_pos[0])

                r_align = 0.5 * math.cos(ag_theta - ang)

                reward = r_dist + r_align

            elif USE_COUPLED == "no_alignment":

                opn = openness(env, int(round(sub_goal[0])), int(round(sub_goal[1])))

                r_open = 0.3 * opn

                reward = r_dist + r_open

            else:

                ang = math.atan2(sub_goal[1]-env.agent_pos[1], sub_goal[0]-env.agent_pos[0])

                r_align = 0.5 * math.cos(ag_theta - ang)

                opn = openness(env, int(round(sub_goal[0])), int(round(sub_goal[1])))

                r_open = 0.3 * opn

                reward = r_dist + r_align + r_open



            if done:

                if info.get("reason")=="goal": reward=50.0

                elif info.get("reason") in ("collision","boundary"): reward=-10.0

        else:

            reward = r_orig



        if new_d < old_d:

            ag_theta = math.atan2(env.agent_pos[1]-old_pos[1], env.agent_pos[0]-old_pos[0])



        tr += reward

        if USE_CAT:

            mem_entry = (np.stack(list(history)).astype(np.float32), action,

                         np.stack(list(history)[1:]+[ns]).astype(np.float32), reward, done)

        else:

            mem_entry = (s.astype(np.float32), action, ns.astype(np.float32), reward, done)

        memory.append(mem_entry)

        if USE_CAT: history.append(ns)

        s = ns; steps += 1



        if len(memory) >= BATCH:

            batch = random.sample(memory, BATCH)

            sb = torch.FloatTensor(np.array([m[0] for m in batch])).to(device)

            ab = torch.LongTensor([m[1] for m in batch]).unsqueeze(1).to(device)

            nsb = torch.FloatTensor(np.array([m[2] for m in batch])).to(device)

            rb = torch.FloatTensor([m[3] for m in batch]).unsqueeze(1).to(device)

            db = torch.FloatTensor([m[4] for m in batch]).unsqueeze(1).to(device)



            curr = policy(sb).gather(1, ab)

            with torch.no_grad():

                tgt = rb + (1-db)*GAMMA*target(nsb).max(1,keepdim=True)[0]

            nn.MSELoss()(curr, tgt).backward(); opt.step(); opt.zero_grad()



    with torch.no_grad():

        for tp, p in zip(target.parameters(), policy.parameters()):

            tp.data.copy_(TAU*p.data+(1-TAU)*tp.data)

    if eps > EPS_END: eps *= EPS_DECAY

    rewards.append(tr)

    successes.append(info.get("reason")=="goal")

    reward_log.append(tr)



    if (ep+1)%200==0:

        avg_r = np.mean(rewards[-200:])

        sr = sum(successes[-200:])/2

        elapsed = time.time()-start_t

        time_per = elapsed/(ep+1)

        remaining = time_per*(EPISODES-ep-1)

        print(f"{NAME} Ep {ep+1}/{EPISODES} | AvgRew: {avg_r:.1f} | SR: {sr:.1f}% | Eps: {eps:.3f} | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s", flush=True)



t = time.time()-start_t; print(f"{NAME} Done! Time: {t:.0f}s", flush=True)



# Save CSV log

np.savetxt(f"{NAME}_reward_log.csv", np.array(reward_log), fmt="%.18e")

print(f"Reward log saved to {NAME}_reward_log.csv", flush=True)



# Eval

print(f"{NAME} Eval (greedy, 200 eps)...", flush=True)

policy.eval(); sg, lg = 0, []

for ep in range(200):

    s = env.reset()

    history = deque([s]*SEQ_LEN, maxlen=SEQ_LEN)

    done, st = False, 0

    while not done and st < MAX_STEPS:

        with torch.no_grad():

            if USE_CAT:

                seq = torch.FloatTensor(np.stack(list(history))).unsqueeze(0).to(device)

                a = policy(seq).max(1)[1].item()

            else:

                st_t = torch.FloatTensor(s).unsqueeze(0).to(device)

                a = policy(st_t).max(1)[1].item()

        ns, _, done, info = env.step(a)

        if USE_CAT: history.append(ns)

        s = ns; st += 1

    if info.get("reason")=="goal": sg+=1; lg.append(st)

avg_len = np.mean(lg) if lg else 0

print(f"{NAME} Greedy: {sg}/200 = {sg/2:.1f}% | AvgLen: {avg_len:.1f}", flush=True)

torch.save(policy.state_dict(), f"dqn_{NAME}_10000.pth")

print(f"{NAME} Model saved.", flush=True)

