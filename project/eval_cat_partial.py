import sys, numpy as np, math, random, torch, torch.nn as nn
from collections import deque
sys.path.insert(0, '.')
from models import CrossAttentionDQN

random.seed(42); np.random.seed(42)
size=20; obstacles=set(); target=int(size*size*0.2)
start,goal=(1,1),(18,18)
while len(obstacles)<target:
    x=random.randint(1,size-2); y=random.randint(1,size-2)
    if (x,y)!=start and (x,y)!=goal and abs(x-start[0])+abs(y-start[1])>2:
        obstacles.add((x,y))
actions=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def get_state(agent):
    dx=(goal[0]-agent[0])/size; dy=(goal[1]-agent[1])/size
    dist=math.hypot(dx*size,dy*size)/size
    obs_f=[]
    for dx_a,dy_a in [(0,-1),(0,1),(-1,0),(1,0)]:
        nx,ny=agent[0]+dx_a,agent[1]+dy_a
        if nx<0 or nx>=size or ny<0 or ny>=size: obs_f.append(1.0)
        elif (nx,ny) in obstacles: obs_f.append(1.0)
        else: obs_f.append(0.0)
    return np.array([dx,dy,dist]+obs_f, dtype=np.float32)

class CAT4(CrossAttentionDQN):
    def __init__(self):
        super().__init__(d_model=64, nhead=4, seq_len=5)
        self.obs_proj = nn.Linear(4, 64)

device=torch.device('cpu')
SEQ_LEN=5; MAX_STEPS=400
net=CAT4().to(device)
sd=torch.load('dqn_CAT_ckpt_1500_partial.pth', map_location='cpu', weights_only=True)
net.load_state_dict(sd); net.eval()

sg=0
for ep in range(100):
    agent=start; done,st=False,0
    s=get_state(agent); h=deque([s]*SEQ_LEN,maxlen=SEQ_LEN)
    while not done and st<MAX_STEPS:
        with torch.no_grad():
            seq=torch.FloatTensor(np.stack(list(h))).unsqueeze(0)
            a=net(seq).max(1)[1].item()
        dx_a,dy_a=actions[a]; nx,ny=agent[0]+dx_a,agent[1]+dy_a
        if nx<0 or nx>=size or ny<0 or ny>=size or (nx,ny) in obstacles:
            done=True
        else:
            agent=(nx,ny); s=get_state(agent); h.append(s)
            if agent==goal: sg+=1; done=True
        st+=1
print(f'CAT (1500ep) Eval: {sg}/100 = {sg}%')
