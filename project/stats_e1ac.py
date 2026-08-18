import numpy as np

def stat(fn):
    d = np.loadtxt(fn)
    best50 = max(np.mean(d[i:i+50]) for i in range(len(d)-50))
    target = 0.95 * best50
    conv = next((i+50 for i in range(50, len(d)-50)
                 if np.mean(d[i:i+50]) >= target), len(d))
    last200 = np.mean(d[-200:])
    return conv, round(best50, 1), round(float(last200), 1)

for fn in [
    "E1_CAT_ASGS_20x20_reward_log.csv",
    "E1_CAT_ASGS_20x20_run2_reward_log.csv",
    "E1_CAT_ASGS_20x20_run3_reward_log.csv",
    "E1_CAT_ASGS_30x30_reward_log.csv",
    "E1_coupled_ASGS_20x20_reward_log.csv",
    "E2_CAT_coupled_20x20_reward_log.csv",
    "E1_coupled_20x20_reward_log.csv",
]:
    print(fn, stat(fn))
