import numpy as np
import math

class DWAPlanner:
    def __init__(self, v_max=1.0, w_max=1.0, v_acc=0.5, w_acc=0.5, dt=0.1, eval_time=1.0):
        self.v_max = v_max
        self.v_min = 0.0
        self.w_max = w_max
        self.v_acc = v_acc
        self.w_acc = w_acc
        self.dt = dt
        self.eval_time = eval_time
        self.sample_num = 10

    def plan(self, start_pos, goal_pos, obstacles, current_v=0.0, current_w=0.0, current_theta=0.0):
        v_samples = np.linspace(self.v_min, self.v_max, self.sample_num)
        w_samples = np.linspace(-self.w_max, self.w_max, self.sample_num)
        
        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0
        
        start_x, start_y = start_pos
        gx, gy = goal_pos
        start_dist = math.hypot(start_x - gx, start_y - gy)

        for v in v_samples:
            for w in w_samples:
                traj = self._simulate_trajectory(start_pos, v, w, current_theta)
                if self._has_collision(traj, obstacles):
                    continue
                score = self._evaluate_traj(traj, goal_pos, obstacles, v, w, start_dist)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
        return best_v, best_w

    def _simulate_trajectory(self, start, v, w, start_theta=0.0):
        traj = [(start[0], start[1])]
        x, y = start
        theta = start_theta
        steps = int(self.eval_time / self.dt)
        for _ in range(steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += w * self.dt
            traj.append((x, y))
        return traj

    def _has_collision(self, traj, obstacles):
        # Check ALL trajectory points for accuracy
        for x, y in traj:
            gx, gy = int(round(x)), int(round(y))
            if gx < 0 or gx >= 15 or gy < 0 or gy >= 15:
                return True
            if (gx, gy) in obstacles:
                return True
        return False

    def _evaluate_traj(self, traj, goal, obstacles, v, w, start_dist):
        if len(traj) < 2:
            return 0.0
        final_x, final_y = traj[-1]
        gx, gy = goal
        end_dist = math.hypot(final_x - gx, final_y - gy)

        improvement = start_dist - end_dist
        heading = max(0.0, improvement / (start_dist + 1e-8))

        clearance = 1.0
        for (ox, oy) in obstacles:
            d = math.hypot(final_x - (ox + 0.5), final_y - (oy + 0.5))
            if d < clearance:
                clearance = d
        clearance = min(clearance / 2.0, 1.0)

        speed = v / self.v_max
        return 0.4 * heading + 0.3 * clearance + 0.3 * speed
