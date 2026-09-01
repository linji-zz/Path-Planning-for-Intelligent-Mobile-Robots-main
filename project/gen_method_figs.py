"""gen_method_figs.py — Draw schematic figures for the method section (第3节 + 4.1).

Usage: python gen_method_figs.py [dpi] [outdir]

Figures (method / setup):
  fig_rl_interaction.png   - 强化学习交互过程图 (3.1)
  fig_dqn_structure.png    - DQN 训练结构图 (3.1)
  fig_cat_structure.png    - CAT-DQN 网络结构图 (3.3)
  fig_lambda_curve.png     - ASGS lambda 曲线 (3.4)
  fig_training_flow.png    - 完整方法训练流程图 (3.5)
  fig_grid_maps.png        - 20x20 / 30x30 栅格地图 (4.1)
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main"
PROJ = os.path.join(ROOT, "project")
sys.path.insert(0, PROJ)

DPI = int(sys.argv[1]) if len(sys.argv) > 1 else 150
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else PROJ
os.makedirs(OUTDIR, exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#dbeafe", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="black", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2, text=None, fs=8, txy=(0, 2)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    if text:
        ax.annotate(text, xy=((x1 + x2) / 2, (y1 + y2) / 2),
                    xytext=txy, textcoords="offset points",
                    ha="center", va="center", fontsize=fs)


def fig_rl_interaction():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.08, 0.35, 0.30, 0.30, "智能体\n(Agent)", fc="#dbeafe", fs=11)
    _box(ax, 0.62, 0.35, 0.30, 0.30, "环境\n(Environment)", fc="#d1fae5", fs=11)
    _arrow(ax, 0.38, 0.62, 0.62, 0.62, "动作 a", fs=9)
    _arrow(ax, 0.62, 0.38, 0.38, 0.38, "状态 s′、奖励 r", fs=9, txy=(0, -14))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_rl_interaction.png"), dpi=DPI)
    plt.close()
    print("Saved fig_rl_interaction.png")


def fig_dqn_structure():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.02, 0.75, 0.22, 0.16, "环境\n(Environment)", fc="#d1fae5", fs=10)
    _box(ax, 0.02, 0.42, 0.22, 0.20, "经验回放池\n(s, a, r, s′)", fc="#fef3c7", fs=10)
    _box(ax, 0.38, 0.72, 0.24, 0.20, "估计网络\nQ(s, a; θ)", fc="#dbeafe", fs=10)
    _box(ax, 0.38, 0.40, 0.24, 0.20, "目标网络\nQ(s′, a′; θ⁻)", fc="#e0e7ff", fs=10)
    _box(ax, 0.74, 0.56, 0.24, 0.20, "损失函数\nMSE(y′ − Q)", fc="#fce7f3", fs=10)
    _box(ax, 0.38, 0.08, 0.24, 0.16, "梯度更新", fc="#f3f4f6", fs=10)
    _arrow(ax, 0.24, 0.80, 0.24, 0.62, None)
    _arrow(ax, 0.13, 0.62, 0.13, 0.42, None) if False else None
    _arrow(ax, 0.24, 0.52, 0.38, 0.62, "采样批量", fs=8)
    _arrow(ax, 0.62, 0.78, 0.74, 0.66, "Q(s,a)", fs=8)
    _arrow(ax, 0.50, 0.50, 0.74, 0.50, "y′ = r + γmaxQ", fs=8, txy=(0, 8))
    _arrow(ax, 0.86, 0.56, 0.50, 0.24, "反向传播", fs=8, txy=(-4, 12))
    _arrow(ax, 0.50, 0.40, 0.50, 0.24, "软更新 τ", fs=8, txy=(6, -12))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_dqn_structure.png"), dpi=DPI)
    plt.close()
    print("Saved fig_dqn_structure.png")


def fig_cat_structure():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.02, 0.60, 0.20, 0.30, "目标分支\n(dx, dy, dist)\n3 维", fc="#dbeafe", fs=10)
    _box(ax, 0.02, 0.15, 0.20, 0.30, "障碍分支\n(obs0~obs7)\n8 维", fc="#d1fae5", fs=10)
    _box(ax, 0.30, 0.62, 0.18, 0.26, "自注意力\n编码", fc="#fef3c7", fs=9)
    _box(ax, 0.30, 0.17, 0.18, 0.26, "自注意力\n编码", fc="#fef3c7", fs=9)
    _box(ax, 0.56, 0.42, 0.22, 0.28, "交叉注意力\n(AG→Query, OBS→K/V)", fc="#fce7f3", fs=9)
    _box(ax, 0.82, 0.42, 0.15, 0.28, "输出层\n8 个 Q 值", fc="#f3f4f6", fs=9)
    _arrow(ax, 0.22, 0.72, 0.30, 0.72, None)
    _arrow(ax, 0.22, 0.32, 0.30, 0.32, None)
    _arrow(ax, 0.48, 0.70, 0.56, 0.60, None)
    _arrow(ax, 0.48, 0.34, 0.56, 0.48, None)
    _arrow(ax, 0.78, 0.56, 0.82, 0.56, None)
    ax.text(0.10, 0.95, "5 步历史状态序列输入", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_cat_structure.png"), dpi=DPI)
    plt.close()
    print("Saved fig_cat_structure.png")


def fig_lambda_curve():
    K = 10000
    k = np.linspace(0, K, 200)
    lam = 5.0 + 10.0 * k / K
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, lam, lw=2, color="tab:blue")
    ax.set_xlabel("训练轮数 k")
    ax.set_ylabel("λ")
    ax.set_ylim(0, 16)
    ax.set_title("ASGS 修正强度 λ 随训练进度线性增强")
    ax.grid(alpha=0.3)
    ax.text(K * 0.02, 12.5, "λ = 5.0 + 10.0·k/K", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_lambda_curve.png"), dpi=DPI)
    plt.close()
    print("Saved fig_lambda_curve.png")


def fig_training_flow():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.36, 0.88, 0.28, 0.10, "环境（栅格地图）", fc="#d1fae5", fs=10)
    _box(ax, 0.36, 0.66, 0.28, 0.10, "状态 s（11 维）", fc="#fef3c7", fs=10)
    _box(ax, 0.36, 0.44, 0.28, 0.10, "CAT-DQN 网络\n输出 8 个 Q 值", fc="#dbeafe", fs=10)
    _box(ax, 0.36, 0.22, 0.28, 0.10, "ASGS 修正\nQ′ = Q − λ·obs", fc="#fce7f3", fs=10)
    _box(ax, 0.36, 0.02, 0.28, 0.10, "动作 a → 环境", fc="#f3f4f6", fs=10)
    _arrow(ax, 0.50, 0.88, 0.50, 0.76, None)
    _arrow(ax, 0.50, 0.66, 0.50, 0.54, None)
    _arrow(ax, 0.50, 0.44, 0.50, 0.32, None)
    _arrow(ax, 0.50, 0.22, 0.50, 0.12, None)
    _arrow(ax, 0.78, 0.07, 0.78, 0.93, "CRF 耦合奖励 r", fs=9, txy=(6, 0))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_training_flow.png"), dpi=DPI)
    plt.close()
    print("Saved fig_training_flow.png")


def fig_grid_maps():
    from env_large import GridEnvLarge
    from env_large30 import GridEnv30
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, env, title in [(axes[0], GridEnvLarge(), "(a) 20×20 地图"),
                           (axes[1], GridEnv30(), "(b) 30×30 地图")]:
        ax.set_xlim(-0.5, env.size - 0.5)
        ax.set_ylim(-0.5, env.size - 0.5)
        for (ox, oy) in env.obstacles:
            ax.add_patch(Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="black", edgecolor="none"))
        ax.plot(env.start[0], env.start[1], "o", color="blue", markersize=8, zorder=5)
        ax.plot(env.goal[0], env.goal[1], "s", color="green", markersize=8, zorder=5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_grid_maps.png"), dpi=DPI)
    plt.close()
    print("Saved fig_grid_maps.png")


if __name__ == "__main__":
    fig_rl_interaction()
    fig_dqn_structure()
    fig_cat_structure()
    fig_lambda_curve()
    fig_training_flow()
    fig_grid_maps()
    print("ALL METHOD FIGURES DONE")
