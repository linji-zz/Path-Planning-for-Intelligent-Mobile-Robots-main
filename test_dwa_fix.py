import sys, math, numpy as np
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

from dwa_env import DWAEnv

params = {"v_max": 2.0, "w_max": 2.0, "v_acc": 1.0, "w_acc": 1.0, "dt": 0.1, "eval_time": 0.5}
env = DWAEnv(dwa_params=params, coupled_reward=False)

# Test: move toward goal (action 7 = (1,1), diagonal right-down)
state = env.reset()
print(f"Start: grid={env.agent_pos}, cont=({env.cont_x:.2f},{env.cont_y:.2f}), theta={env.cont_theta:.2f}")
print(f"Goal: {env.goal}")
print(f"Sub_steps per DQN step: {max(1, int(env.dwa.eval_time/env.dwa.dt))}")

# Take 20 steps always moving diagonal toward goal (action 7 = 1,1)
for i in range(30):
    ns, r, done, info = env.step(7)  # action 7 = (1,1) = diagonal SE
    gx, gy = env._get_grid_pos()
    d = math.hypot(gx-env.goal[0], gy-env.goal[1])
    print(f"Step {i+1}: pos=({env.cont_x:.2f},{env.cont_y:.2f}), grid={gx,gy}, "+
          f"theta={env.cont_theta:.2f}, dist={d:.2f}, reward={r:.2f}, done={done} ({info.get('reason')})")
    if done:
        break

# Quick check: run a random policy for 10 episodes
print("\n--- Random Policy Test (10 episodes) ---")
for ep in range(10):
    state = env.reset()
    d = False
    steps = 0
    while not d and steps < 100:
        a = np.random.randint(8)
        _, _, d, info = env.step(a)
        steps += 1
    gx, gy = env._get_grid_pos()
    dist = math.hypot(gx-env.goal[0], gy-env.goal[1])
    print(f"Ep {ep+1}: {steps} steps, end=({gx},{gy}), reason={info.get('reason')}, dist={dist:.1f}")
