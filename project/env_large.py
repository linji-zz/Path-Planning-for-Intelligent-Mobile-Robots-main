import numpy as np, math, random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OBSTACLES = set([
    # 左上方斜墙 (阻挡直接对角线路径)
    (5,2),(5,3),(5,4),(5,5),
    (2,5),(3,5),(4,5),
    # 中央屏障 (需绕行)
    (10,6),(10,7),(10,8),(10,9),(10,10),(10,11),
    (6,10),(7,10),(8,10),(9,10),
    # 右下障碍群
    (14,12),(14,13),(14,14),(14,15),
    (12,14),(13,14),
    # 下方集群
    (17,13),(17,14),(17,15),(17,16),(17,17),
    (13,17),(14,17),(15,17),(16,17),
    # 散落障碍物
    (3,10),(4,10),
    (8,3),(8,4),
    (12,5),(13,5),
    (4,13),(4,14),
    (7,15),(8,15),
    (15,7),(15,8),
    (18,10),(18,11),
    (10,3),(11,3),
    (2,13),(2,14),(2,15),
    (7,18),(8,18),(9,18),
])

class GridEnvLarge:
    def set_sparse(self, val=True):
        self.sparse = val

    def __init__(self, seed=42):
        self.size = 20
        self.obstacles = OBSTACLES
        self.start = (1, 1)
        self.goal = (18, 18)
        self.actions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self.action_dim = 8
        random.seed(seed); np.random.seed(seed)
        self.sparse = False
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
        return np.array([dx,dy,dist]+obs_flags, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx,dy = self.actions[action]
        old_pos = self.agent_pos
        new_x = old_pos[0]+dx; new_y = old_pos[1]+dy
        if new_x<0 or new_x>=self.size or new_y<0 or new_y>=self.size:
            self.done = True
            return self._get_state(), -1.0 if self.sparse else -10.0, True, {"reason":"boundary"}
        if (new_x,new_y) in self.obstacles:
            self.done = True
            return self._get_state(), -1.0 if self.sparse else -10.0, True, {"reason":"collision"}
        self.agent_pos = (new_x,new_y)
        old_d = math.hypot(old_pos[0]-self.goal[0], old_pos[1]-self.goal[1])
        new_d = math.hypot(self.agent_pos[0]-self.goal[0], self.agent_pos[1]-self.goal[1])
        reward = (old_d-new_d)*2.0
        if self.agent_pos == self.goal:
            self.done = True
            return self._get_state(), 10.0 if self.sparse else 50.0, True, {"reason":"goal"}
        if self.steps >= 400:
            self.done = True
            return self._get_state(), reward if not self.sparse else 0.0, True, {"reason":"max_steps"}
        if self.sparse:
            return self._get_state(), 0.0, False, {}
        return self._get_state(), reward, False, {}

    def render(self, path=None, ax=None, title=""):
        if ax is None: fig,ax = plt.subplots(figsize=(8,8))
        ax.clear()
        ax.set_xlim(-0.5,self.size-0.5); ax.set_ylim(-0.5,self.size-0.5)
        for i in range(self.size+1):
            ax.axhline(y=i-0.5,color='gray',linewidth=0.5,alpha=0.3)
            ax.axvline(x=i-0.5,color='gray',linewidth=0.5,alpha=0.3)
        for (ox,oy) in self.obstacles:
            ax.add_patch(Rectangle((ox-0.5,oy-0.5),1,1,facecolor='black',edgecolor='none'))
        if path:
            path_arr = np.array(path)
            ax.plot(path_arr[:,0], path_arr[:,1], 'r-', linewidth=2, alpha=0.8)
        ax.plot(self.start[0],self.start[1],'bo',markersize=12,label='Start')
        ax.plot(self.goal[0],self.goal[1],'gs',markersize=12,label='Goal')
        ax.set_title(title); ax.legend(); ax.set_aspect('equal')
        plt.draw(); plt.pause(0.01)

if __name__=="__main__":
    env = GridEnvLarge()
    fig,ax = plt.subplots(figsize=(8,8))
    env.render(ax=ax, title="20x20 复杂地图")
    plt.savefig("map_20x20.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"20x20, 障碍物: {len(env.obstacles)}个, 起点={env.start}, 终点={env.goal}")
    print(f"从(1,1)到(18,18)对角线最优步数 = {max(17,17)} 步（无障碍时）")
