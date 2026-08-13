"""gen_summary_table.py - Generate complete comparison table for all methods."""
import numpy as np, os

BASE = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main"
LOGDIR = os.path.join(BASE, "project")

def find_log(name):
    p = os.path.join(LOGDIR, name)
    if os.path.exists(p):
        return p
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        return p
    return None

def log_stats(logf):
    d = np.loadtxt(find_log(logf))
    best50 = max(np.mean(d[i:i+50]) for i in range(len(d)-50))
    last200 = np.mean(d[-200:])
    target = 0.95*best50
    conv = len(d)
    for i in range(50, len(d)-50):
        if np.mean(d[i:i+50]) >= target:
            conv = i+50; break
    return conv, round(best50,1), round(last200,1)

# Method: (logs list, times list, SRs list, greedy success, path len)
methods20 = {
    "S0 Sparse": (["S0_sparse_baseline_20x20_reward_log.csv","S0_sparse_baseline_20x20_run2_reward_log.csv","S0_sparse_baseline_20x20_run3_reward_log.csv"], [748,817,829], [83.0,77.0,82.0], "100%", "22.7"),
    "E1 Coupled": (["E1_coupled_20x20_reward_log.csv","E1_coupled_20x20_run2_reward_log.csv","E1_coupled_20x20_run3_reward_log.csv"], [354,342,342], [85.0,84.5,85.5], "100%", "22.0"),
    "E1A Coupled+ASGS": (["E1_coupled_ASGS_20x20_reward_log.csv","E1_coupled_ASGS_20x20_run2_reward_log.csv","E1_coupled_ASGS_20x20_run3_reward_log.csv"], [462,457,410], [98.5,96.5,98.0], "100%", "22.0"),
    "DDQN": (["DDQN_20x20_coupled_reward_log.csv","DDQN_20x20_coupled_run2_reward_log.csv","DDQN_20x20_coupled_run3_reward_log.csv"], [347,315,327], [83.0,81.5,86.0], "100%", "22.0"),
    "Dueling": (["DUELING_20x20_coupled_reward_log.csv","DUELING_20x20_coupled_run2_reward_log.csv","DUELING_20x20_coupled_run3_reward_log.csv"], [384,350,455], [80.0,82.5,81.0], "100%", "22.0"),
    "PER": (["PER_20x20_coupled_reward_log.csv","PER_20x20_coupled_run2_reward_log.csv","PER_20x20_coupled_run3_reward_log.csv"], [724,775,809], [84.5,86.5,83.0], "100%", "22.0"),
}

methods30 = {
    "S0 Sparse": (["S0_sparse_baseline_30x30_reward_log.csv","S0_sparse_baseline_30x30_run2_reward_log.csv","S0_sparse_baseline_30x30_run3_reward_log.csv"], [929,1270,676], [59.0,0.0,81.5], "66.7%", "36.0"),
    "E1 Coupled": (["E1_coupled_30x30_reward_log.csv","E1_coupled_30x30_run2_reward_log.csv","E1_coupled_30x30_run3_reward_log.csv"], [522,503,524], [81.0,75.5,69.0], "100%", "33.7"),
    "E1A Coupled+ASGS": (["E1_coupled_ASGS_30x30_reward_log.csv","E1_coupled_ASGS_30x30_run2_reward_log.csv","E1_coupled_ASGS_30x30_run3_reward_log.csv"], [640,771,620], [97.5,96.5,96.5], "100%", "33.7"),
    "DDQN": (["DDQN_30x30_coupled_reward_log.csv","DDQN_30x30_coupled_run2_reward_log.csv","DDQN_30x30_coupled_run3_reward_log.csv"], [693,649,630], [81.0,81.5,87.0], "100%", "33.0"),
    "Dueling": (["DUELING_30x30_coupled_reward_log.csv","DUELING_30x30_coupled_run2_reward_log.csv","DUELING_30x30_coupled_run3_reward_log.csv"], [762,705,683], [77.5,84.5,85.0], "100%", "33.0"),
    "PER": (["PER_30x30_coupled_reward_log.csv","PER_30x30_coupled_run2_reward_log.csv","PER_30x30_coupled_run3_reward_log.csv"], [1057,1191,968], [79.0,83.0,82.5], "100%", "33.0"),
}

def summarize(methods, title):
    print("="*110)
    print(title)
    print("="*110)
    print("{:<22} {:<12} {:<10} {:<12} {:<12} {:<10}".format(
        "Method","TrainTime(s)","ConvEp","Best50","TrainSR","PathLen"))
    print("-"*110)
    for name, (logs, times, srs, gsr, plen) in methods.items():
        convs, bests = [], []
        for lf in logs:
            p = find_log(lf)
            if p and os.path.exists(p):
                c, b, _ = log_stats(lf)
                convs.append(c); bests.append(b)
        cm = int(np.mean(convs)) if convs else 0
        cs = int(np.std(convs)) if convs else 0
        bm = round(np.mean(bests),1) if bests else 0
        bs = round(np.std(bests),1) if bests else 0
        tm = int(np.mean(times))
        sm = round(np.mean(srs),1); ss = round(np.std(srs),1)
        print("{:<22} {:<12} {:<10} {:<12} {:<12} {:<10}".format(
            name, str(tm), f"{cm}+/-{cs}", f"{bm}+/-{bs}", f"{sm}+/-{ss}%", plen))
    print()

def per_seed(methods, title):
    print("="*110)
    print(title + " (per-seed detail)")
    print("="*110)
    print("{:<22} {:<8} {:<12} {:<10} {:<12} {:<10} {:<10}".format(
        "Method","Seed","TrainTime(s)","ConvEp","Best50","Last200","TrainSR"))
    print("-"*110)
    for name, (logs, times, srs, gsr, plen) in methods.items():
        for i, lf in enumerate(logs):
            p = find_log(lf)
            if not p or not os.path.exists(p):
                continue
            c, b, l = log_stats(lf)
            print("{:<22} {:<8} {:<12} {:<10} {:<12} {:<10} {:<10}".format(
                name, f"seed{i+1}", times[i], c, b, l, f"{srs[i]}%"))
    print()

summarize(methods20, "20x20 COMPLETE COMPARISON TABLE (3 seeds)")
summarize(methods30, "30x30 COMPLETE COMPARISON TABLE (3 seeds)")
per_seed(methods20, "20x20")
per_seed(methods30, "30x30")
