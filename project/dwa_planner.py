import numpy as np
import math

class DWAPlanner:
    def __init__(self, v_max=1.0, w_max=1.0, v_acc=0.5, w_acc=0.5, dt=0.1, eval_time=1.0):
        self.v_max = v_max
        self.v_min = -v_max
        self.w_max = w_max
        self.v_acc = v_acc
        self.w_acc = w_acc
        self.dt = dt
        self.eval_time = eval_time
        self.sample_num = 10  # 从20减到10

    def plan(self, start_pos, goal_pos, obstacles, current_v=0.0, current_w=0.0):
        # 速度采样
        v_samples = np.linspace(
            max(self.v_min, current_v - self.v_acc * self.dt),
            min(self.v_max, current_v + self.v_acc * self.dt),
            self.sample_num
        )
        w_samples = np.linspace(
            max(-self.w_max, current_w - self.w_acc * self.dt),
            min(self.w_max, current_w + self.w_acc * self.dt),
            self.sample_num
        )

        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0

        for v in v_samples:
            for w in w_samples:
                traj = self._simulate_trajectory(start_pos, v, w)
                if self._has_collision(traj, obstacles):
                    continue
                score = self._evaluate_traj(traj, goal_pos, obstacles, v, w)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return best_v, best_w

    def _simulate_trajectory(self, start, v, w):
        traj = []
        x, y = start
        theta = 0.0
        steps = int(self.eval_time / self.dt)
        for _ in range(steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += w * self.dt
            traj.append((x, y))
        return traj

    def _has_collision(self, traj, obstacles):
        # 只检查奇数次点，减少计算
        for i in range(0, len(traj), 2):
            x, y = traj[i]
            if not (0 <= x < 25 and 0 <= y < 25):
                return True
            if (int(round(x)), int(round(y))) in obstacles:
                return True
        return False

    def _evaluate_traj(self, traj, goal, obstacles, v, w):
        if len(traj) == 0:
            return 0.0
        final_x, final_y = traj[-1]
        gx, gy = goal
        dx, dy = gx - final_x, gy - final_y
        dist = math.hypot(dx, dy)

        # Heading
        heading = 0.0
        if len(traj) > 1 and dist > 0:
            last_dx = traj[-1][0] - traj[-2][0]
            last_dy = traj[-1][1] - traj[-2][1]
            if math.hypot(last_dx, last_dy) > 0:
                heading = (last_dx*dx + last_dy*dy) / (math.hypot(last_dx, last_dy) * dist)
                heading = max(0.0, heading)

        # Clearance
        clearance = float('inf')
        for (x, y) in traj:
            for (ox, oy) in obstacles:
                d = math.hypot(x - ox, y - oy)
                if d < clearance:
                    clearance = d
        clearance = min(clearance / 3.0, 1.0)

        # Speed
        speed = v / self.v_max if v > 0 else 0.0

        return 0.4 * heading + 0.4 * clearance + 0.2 * speed