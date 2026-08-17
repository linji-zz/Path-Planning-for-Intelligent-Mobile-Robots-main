import sys
import numpy as np
import torch
import torch.nn as nn
from collections import deque

sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

MAPSIZE = int(sys.argv[1])
MODEL = sys.argv[2]
USE_CAT = sys.argv[3].lower() in ("1", "true", "cat")
USE_ASGS = sys.argv[4].lower() in ("1", "true", "asgs")
EPISODES = int(sys.argv[5]) if len(sys.argv) > 5 else 200

if MAPSIZE == 20:
    from env_large import GridEnvLarge as Env
    env = Env()
    MAX_STEPS = 400
else:
    from env_large30 import GridEnv30 as Env
    env = Env()
    MAX_STEPS = 500

from models import CrossAttentionDQN

def make_mlp():
    return nn.Sequential(nn.Linear(11, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, 8))

policy = CrossAttentionDQN() if USE_CAT else make_mlp()
policy.load_state_dict(torch.load(MODEL, map_location="cpu"))
policy.eval()

SEQ_LEN = 5 if USE_CAT else 1
sg, lg = 0, []
reasons = {}
for _ in range(EPISODES):
    s = env.reset()
    history = deque([s] * SEQ_LEN, maxlen=SEQ_LEN)
    done, st = False, 0
    while not done and st < MAX_STEPS:
        with torch.no_grad():
            if USE_CAT:
                seq = torch.FloatTensor(np.stack(list(history))).unsqueeze(0)
                q = policy(seq)
            else:
                q = policy(torch.FloatTensor(s).unsqueeze(0))
            if USE_ASGS:
                lam = 15.0
                q = q - torch.FloatTensor(s[3:11]).unsqueeze(0) * lam
            a = q.max(1)[1].item()
        ns, _, done, info = env.step(a)
        if USE_CAT:
            history.append(ns)
        s = ns
        st += 1
    reasons[info.get("reason", "?")] = reasons.get(info.get("reason", "?"), 0) + 1
    if info.get("reason") == "goal":
        sg += 1
        lg.append(st)

avg_len = np.mean(lg) if lg else 0
print(f"Greedy({USE_ASGS}): {sg}/{EPISODES} = {sg/EPISODES*100:.1f}% | AvgLen: {avg_len:.1f} | reasons: {reasons}")
