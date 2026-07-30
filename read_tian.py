import sys
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\.venv\Lib\site-packages")
from pypdf import PdfReader
import os

path = r"C:\Users\CAIHUI\Desktop\dqn\基于改进DQN的移动机器人避障路径规划_田箫源.pdf"
reader = PdfReader(path)
full = ""
for page in reader.pages:
    full += page.extract_text() or ""

segments = full.replace("\n", " ").split("。")
for i, s in enumerate(segments):
    s = s.strip()
    if len(s) > 25 and any(k in s for k in ["实验","仿真","环境","结果","对比","20","障碍物","路径","指标","参数","地图","场景","训练时间","成功率"]):
        clean = "".join(c for c in s if ord(c) < 128 or ord(c) > 1279)
        if len(clean) > 30:
            print(f"---")
            print(clean[:300])
