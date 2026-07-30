import sys, os
sys.path.insert(0, 'C:\\Users\\CAIHUI\\Path-Planning-for-Intelligent-Mobile-Robots-main\\.venv\\Lib\\site-packages')
from pypdf import PdfReader

dqn_dir = 'C:\\Users\\CAIHUI\\Desktop\\dqn'
files = sorted([f for f in os.listdir(dqn_dir) if f.endswith('.pdf')])

for fname in files:
    path = os.path.join(dqn_dir, fname)
    try:
        reader = PdfReader(path)
        text = ''
        for page in reader.pages[:3]:
            text += page.extract_text() or ''
        text = text.replace('\n', ' ')[:1500]
        print('=' * 60)
        print(f'FILE: {fname[:50]}')
        print('=' * 60)
        print(text[:800])
        print()
    except Exception as e:
        print(f'ERROR: {fname[:40]} - {str(e)[:50]}')
