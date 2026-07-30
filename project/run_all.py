import sys, time, numpy as np, importlib

EXPERIMENTS = {
    1: ("DQN (baseline)", "train_env1"),
    2: ("DQN + Coupled", "train_coupled"),
    3: ("DQN + DWA", "train_dqn_dwa"),
    4: ("DQN + Coupled + DWA", "train_dqn_coupled_dwa"),
}

def main():
    quick = "--quick" in sys.argv
    results = {}
    for eid, (name, fname) in EXPERIMENTS.items():
        print("\n" + "="*60)
        print(f"Running Exp {eid}: {name}")
        print("="*60)
        mod = importlib.import_module(fname)
        if quick:
            mod.EPISODES = 500
            print(f"  (quick mode: EPISODES=500)")
        t0 = time.time()
        m = mod.train()
        m["time"] = time.time() - t0
        results[eid] = m

    if results:
        print("\n" + "="*80)
        print("COMPARISON TABLE (fill in after all experiments complete)")
        print("="*80)
        h = f"{'Experiment':25s} {'SR(%)':>7s} {'AvgLen':>8s}"
        h += f" {'RewConv':>10s} {'DistConv':>10s} {'Min':>8s}"
        print(h)
        print("-"*80)
        for eid in sorted(results.keys()):
            m = results[eid]
            name, _ = EXPERIMENTS[eid]
            sr = m.get("success_rate", 0) or 0
            al = m.get("avg_length", 0) or 0
            rc = str(m.get("reward_conv", "N/A"))
            dc = str(m.get("dist_conv", "N/A"))
            tm = m.get("train_time", m.get("time", 0)) / 60
            print(f"{name:25s} {sr:7.1f}% {al:8.2f} {rc:>10s} {dc:>10s} {tm:7.1f}")
        print("-"*80)

if __name__ == "__main__":
    main()
