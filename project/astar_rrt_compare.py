"""astar_rrt_compare.py - A* and RRT comparison on grid maps.
Usage: python astar_rrt_compare.py
Runs A* (optimal) and RRT (stochastic, multiple runs) on 20x20 and 30x30 maps.
Outputs path steps, path length, and turn counts for comparison.
"""
import sys, math, random, heapq
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\project")

ACTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def astar(env):
    """A* search on the grid, 8-directional moves. Returns path list of (x,y)."""
    start, goal = env.start, env.goal
    size = env.size
    obstacles = env.obstacles
    def h(a, b):
        dx = abs(a[0]-b[0]); dy = abs(a[1]-b[1])
        return max(dx, dy)  # diagonal distance (Chebyshev)
    open_set = [(h(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for dx, dy in ACTIONS:
            nxt = (current[0]+dx, current[1]+dy)
            if nxt[0] < 0 or nxt[0] >= size or nxt[1] < 0 or nxt[1] >= size:
                continue
            if nxt in obstacles:
                continue
            cost = 1.0 if (dx == 0 or dy == 0) else math.sqrt(2.0)
            tentative = g + cost
            if tentative < g_score.get(nxt, float('inf')):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + h(nxt, goal)
                heapq.heappush(open_set, (f, tentative, nxt))
    return None

def rrt(env, max_iter=20000, goal_tolerance=0.5):
    """Simple RRT on the grid (continuous relaxation). Returns path list."""
    start, goal = env.start, env.goal
    size = env.size
    obstacles = env.obstacles
    # sample points on grid with random offsets
    tree = {start: None}
    for _ in range(max_iter):
        # sample random point
        if random.random() < 0.1:
            sample = (goal[0] + random.uniform(-0.4, 0.4), goal[1] + random.uniform(-0.4, 0.4))
        else:
            sample = (random.uniform(0, size-1), random.uniform(0, size-1))
        # nearest node
        nearest = min(tree.keys(), key=lambda p: (p[0]-sample[0])**2 + (p[1]-sample[1])**2)
        # extend toward sample by small step
        dx = sample[0] - nearest[0]; dy = sample[1] - nearest[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        step = 0.5
        new_x = nearest[0] + dx/dist*step
        new_y = nearest[1] + dy/dist*step
        # check if new point collides (snap to nearest cell)
        cx, cy = int(round(new_x)), int(round(new_y))
        if cx < 0 or cx >= size or cy < 0 or cy >= size:
            continue
        if (cx, cy) in obstacles:
            continue
        new_node = (new_x, new_y)
        if new_node in tree:
            continue
        tree[new_node] = nearest
        # check goal reached
        if math.hypot(new_x-goal[0], new_y-goal[1]) < goal_tolerance:
            # reconstruct path
            path = [(goal[0], goal[1])]
            node = new_node
            while node is not None:
                path.append((node[0], node[1]))
                node = tree[node]
            path.reverse()
            return path
    return None

def path_stats(path):
    """Compute steps, total length, and turn count from a path."""
    if path is None or len(path) < 2:
        return 0, 0, 0
    steps = len(path) - 1
    length = 0.0
    turns = 0
    prev_dir = None
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        length += math.hypot(dx, dy)
        cur_dir = (1 if dx > 0.5 else (-1 if dx < -0.5 else 0),
                   1 if dy > 0.5 else (-1 if dy < -0.5 else 0))
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return steps, round(length, 2), turns

def main():
    from env_large import GridEnvLarge
    from env_large30 import GridEnv30

    results = []
    for env_cls, name in [(GridEnvLarge, "20x20"), (GridEnv30, "30x30")]:
        env = env_cls()
        print("="*70)
        print(f"Map: {name} | Start: {env.start} | Goal: {env.goal} | Obstacles: {len(env.obstacles)}")
        print("="*70)

        # A*
        path = astar(env)
        steps, length, turns = path_stats(path)
        print(f"A*      : Steps={steps}  Length={length}  Turns={turns}")
        a_res = (name, "A*", steps, length, turns)

        # RRT (multiple runs, take best and average)
        best = None
        avg_steps, avg_len, avg_turns, successes = 0, 0, 0, 0
        for trial in range(20):
            p = rrt(env)
            if p:
                s, l, t = path_stats(p)
                avg_steps += s; avg_len += l; avg_turns += t; successes += 1
                if best is None or s < best[1]:
                    best = (s, l, t)
        if successes > 0:
            avg_steps /= successes; avg_len /= successes; avg_turns /= successes
            print(f"RRT     : Best Steps={best[0]}  Best Len={best[1]}  Best Turns={best[2]} | Avg(success {successes}/20): Steps={avg_steps:.1f} Len={avg_len:.1f} Turns={avg_turns:.1f}")
            r_res = (name, "RRT", round(avg_steps,1), round(avg_len,1), round(avg_turns,1))
        else:
            print("RRT     : FAILED to find path in 20 trials")
            r_res = (name, "RRT", "-", "-", "-")
        results.append(a_res); results.append(r_res)
        print()

    print("="*70)
    print("SUMMARY (compare with DQN methods)")
    print("="*70)
    for name, method, steps, length, turns in results:
        print(f"{name} {method:<6}: Steps={steps}  Length={length}  Turns={turns}")

if __name__ == "__main__":
    random.seed(42)
    main()
