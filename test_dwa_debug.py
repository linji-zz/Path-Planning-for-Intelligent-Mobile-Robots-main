import sys, math, numpy as np
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

from dwa_env import DWAEnv
from dwa_planner import DWAPlanner

# Test DWA planner directly
planner = DWAPlanner(v_max=2.0, w_max=2.0, v_acc=1.0, w_acc=1.0, dt=0.1, eval_time=0.5)

start = (1.5, 1.5)
goal = (2.5, 2.5)
obstacles = set()

v, w = planner.plan(start, goal, obstacles, current_v=0.0, current_w=0.0, current_theta=0.0)
print(f"From {(1.5,1.5)} to {(2.5,2.5)}: v={v:.3f}, w={w:.3f}")

# Try more specific velocities
traj_good = planner._simulate_trajectory(start, 1.5, 0.7, 0.0)
traj_slow = planner._simulate_trajectory(start, 0.3, 0.3, 0.0)
traj_fast = planner._simulate_trajectory(start, 2.0, 0.0, 0.0)

sd = math.hypot(1.5-2.5, 1.5-2.5)
for name, traj, v, w in [("fast(2,0)", traj_fast, 2.0, 0.0), ("curve(1.5,0.7)", traj_good, 1.5, 0.7), ("slow(0.3,0.3)", traj_slow, 0.3, 0.3)]:
    end = traj[-1]
    ed = math.hypot(end[0]-2.5, end[1]-2.5)
    imp = sd - ed
    h = max(0.0, imp/(sd+1e-8))
    
    cl = 999.0
    for ox, oy in obstacles:
        for px, py in traj:
            d = math.hypot(px-(ox+0.5), py-(oy+0.5))
            if d < cl: cl = d
    cl = min(cl/2.0, 1.0) if cl < 999 else 1.0
    
    sp = v / 2.0
    score = 0.4*h + 0.4*cl + 0.2*sp
    print(f"  {name}: end=({end[0]:.3f},{end[1]:.3f}), ed={ed:.3f}, h={h:.3f}, cl={cl:.3f}, sp={sp:.3f}, score={score:.3f}")

# Now check with ALL obstacles
print()
obstacles_full = set([(1,13), (0,10), (1,10), (2,10), (1,7), (2,7), (3,7), (4,7), 
    (3,0), (3,1), (3,2), (3,3), (4,2), (5,2), (6,2),
    (9,2), (9,3), (9,4), (7,7), (8,7), (9,7), (10,7), (7,2), (7,3), (7,4),
    (12,5), (13,5), (14,5), (12,8), (13,8), (14,8),
    (11,11), (11,12), (11,13), (11,14)])

v2, w2 = planner.plan(start, goal, obstacles_full, current_v=0.0, current_w=0.0, current_theta=0.0)
print(f"With ALL obstacles: v={v2:.3f}, w={w2:.3f}")

# Check closest obstacle for the fast trajectory
traj = traj_good
min_d = 999
for (ox, oy) in obstacles_full:
    for (px, py) in traj:
        d = math.hypot(px-(ox+0.5), py-(oy+0.5))
        if d < min_d: min_d = d
print(f"Closest obstacle to curve traj: {min_d:.3f}")
