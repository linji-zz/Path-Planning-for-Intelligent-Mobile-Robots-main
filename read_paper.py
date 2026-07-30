import sys, re, os
sys.path.insert(0, r"C:\Users\CAIHUI\Path-Planning-for-Intelligent-Mobile-Robots-main\.venv\Lib\site-packages")
from pypdf import PdfReader

# Find the file
dqn_dir = r"C:\Users\CAIHUI\Desktop\dqn"
files = [f for f in os.listdir(dqn_dir) if f.endswith(".pdf") and ("温" in f or "融合" in f)]
if not files:
    files = [f for f in os.listdir(dqn_dir) if f.endswith(".pdf") and "改进融合" in f]

path = os.path.join(dqn_dir, files[0])
print(f"Reading: {files[0]}", flush=True)

reader = PdfReader(path)
text = ""
for page in reader.pages:
    text += page.extract_text() or ""
text = re.sub(r"\s+", " ", text)

# Find experiment/simulation sections
import re as re2
for match in re2.finditer(r"(?:实验|仿真|地图|环境|参数|结果)[^。]*。", text):
    print(match.group())
    print("---")
