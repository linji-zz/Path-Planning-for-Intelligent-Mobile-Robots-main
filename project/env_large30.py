import numpy as np, math, random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 30x30 密集障碍物地图
# 特点：障碍物占约 15%，有多条可选路径
# 起始 (2,2)，终点 (28,28)
# 直接对角线被多层屏障挡住，必须绕行
# 这种地图下随机探索会频繁撞墙，ASGS能减少无效碰撞

def gen_map(seed=42):
    random.seed(seed); np.random.seed(seed)
    obstacles = set()
    size = 30
    
    # 1. 三层屏障挡住直接对角线
    # 第一层：斜墙 (5-10, 5-10)区域
    for i in range(5, 11):
        obstacles.add((i, 10))
    for i in range(5, 11):
        obstacles.add((10, i))
    
    # 第二层：中央横墙
    for i in range(14, 22):
        obstacles.add((i, 15))
    for i in range(12, 16):
        obstacles.add((20, i))
    
    # 第三层：右下屏障
    for i in range(20, 27):
        obstacles.add((i, 22))
    for i in range(20, 25):
        obstacles.add((25, i))
    
    # 2. 随机补充障碍物至约 15% 密度
    target = int(size * size * 0.15)
    while len(obstacles) < target:
        x = random.randint(1, size-2)
        y = random.randint(1, size-2)
        if (x,y) not in obstacles and (x,y) != (2,2) and (x,y) != (28,28):
            # 不堵死起点和终点附近 2 格
            if abs(x-2)<=2 and abs(y-2)<=2: continue
            if abs(x-28)<=2 and abs(y-28)<=2: continue
            obstacles.add((x,y))
    
    return obstacles

OBSTACLES = gen_map()

class GridEnv30:
    def set_sparse(self, val=True):
        self.sparse = val

    def set_noise(self, level=0.2):
        self.noise_level = level

    def __init__(self):
        self.size = 30
        self.obstacles = OBSTACLES
        self.start = (2, 2)
        self.goal = (28, 28)
        self.actions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self.action_dim = 8
        self.sparse = False
        self.noise_level = 0.0
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        dx = (self.goal[0]-self.agent_pos[0])/self.size
        dy = (self.goal[1]-self.agent_pos[1])/self.size
        dist = math.hypot(dx*self.size, dy*self.size)/self.size
        obs_flags = []
        x,y = self.agent_pos
        for dx_a,dy_a in self.actions:
            nx,ny = x+dx_a, y+dy_a
            if nx<0 or nx>=self.size or ny<0 or ny>=self.size: obs_flags.append(1.0)
            elif (nx,ny) in self.obstacles: obs_flags.append(1.0)
            else: obs_flags.append(0.0)
        if self.noise_level > 0:
            for i in range(len(obs_flags)):
                if random.random() < self.noise_level:
                    obs_flags[i] = 1.0 - obs_flags[i]
        return np.array([dx,dy,dist]+obs_flags, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx,dy = self.actions[action]
        old_pos = self.agent_pos
        new_x = old_pos[0]+dx; new_y = old_pos[1]+dy
        if new_x<0 or new_x>=self.size or new_y<0 or new_y>=self.size:
            self.done = True; return self._get_state(), -1.0 if self.sparse else -10.0, True, {"reason":"boundary"}
        if (new_x,new_y) in self.obstacles:
            self.done = True; return self._get_state(), -1.0 if self.sparse else -10.0, True, {"reason":"collision"}
        self.agent_pos = (new_x,new_y)
        old_d = math.hypot(old_pos[0]-self.goal[0], old_pos[1]-self.goal[1])
        new_d = math.hypot(self.agent_pos[0]-self.goal[0], self.agent_pos[1]-self.goal[1])
        reward = (old_d-new_d)*2.0
        if self.agent_pos == self.goal:
            self.done = True; return self._get_state(), 10.0 if self.sparse else 50.0, True, {"reason":"goal"}
        if self.sparse:
            return self._get_state(), 0.0, False, {}
        if self.steps >= 500:
            self.done = True; return self._get_state(), reward, True, {"reason":"max_steps"}
        return self._get_state(), reward, False, {}

    def render(self, path=None, ax=None, title=""):
        if ax is None: fig,ax = plt.subplots(figsize=(12,12))
        ax.clear()
        ax.set_xlim(-0.5,self.size-0.5); ax.set_ylim(-0.5,self.size-0.5)
        for i in range(self.size+1):
            ax.axhline(y=i-0.5,color='gray',linewidth=0.3,alpha=0.3)
            ax.axvline(x=i-0.5,color='gray',linewidth=0.3,alpha=0.3)
        for (ox,oy) in self.obstacles:
            ax.add_patch(Rectangle((ox-0.5,oy-0.5),1,1,facecolor='black',edgecolor='none'))
        if path:
            path_arr = np.array(path)
            if len(path_arr)>1:
                ax.plot(path_arr[:,0],path_arr[:,1],'r-',linewidth=2,alpha=0.8)
        ax.plot(self.start[0],self.start[1],'bo',markersize=10,label='Start')
        ax.plot(self.goal[0],self.goal[1],'gs',markersize=10,label='Goal')
        ax.set_title(title); ax.legend(); ax.set_aspect('equal')
        plt.draw(); plt.pause(0.01)

if __name__=="__main__":
    env = GridEnv30()
    fig,ax = plt.subplots(figsize=(12,12))
    env.render(ax=ax, title="30x30 Dense Obstacle Map")
    plt.savefig("map_30x30.png", dpi=150, bbox_inches="tight")
    print(f"30x30, 障碍物: {len(env.obstacles)}个, 起点={env.start}, 终点={env.goal}")
