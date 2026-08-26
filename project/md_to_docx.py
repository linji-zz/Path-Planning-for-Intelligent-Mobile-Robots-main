"""md_to_docx.py — Convert paper_draft.md to paper_draft.docx (Chinese paper).

Handles headings (#/##/###/####), markdown tables, bullet lists, inline bold,
and embeds the referenced figures (fig_*.png) at their citation points with
captions. Body = 宋体, headings = 黑体, A4 page.
"""
import os, re
from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "paper_draft.md")
OUT = os.path.join(BASE, "paper_draft.docx")

FIG_CAPTIONS = {
    "fig_paths_20x20_all.png": "图1 20×20地图各方法路径对比",
    "fig_reward_20x20_all.png": "图2 20×20地图奖励收敛曲线",
    "fig_paths_30x30_all.png": "图3 30×30地图各方法路径对比",
    "fig_reward_30x30_all.png": "图4 30×30地图奖励收敛曲线",
    "fig_trad_compare_20x20.png": "图5 20×20地图与传统算法对比",
    "fig_trad_compare_30x30.png": "图6 30×30地图与传统算法对比",
    "fig_noise_robustness.png": "图7 噪声鲁棒性对比",
}
FIG_RE = re.compile(r"fig_[A-Za-z0-9_]+\.png")


def set_font(run, name, size, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), name)


def add_inline(paragraph, text, name="宋体", size=12):
    """Add text with **bold** inline formatting."""
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        run = paragraph.add_run(part)
        set_font(run, name, size, bold=(i % 2 == 1))
    return paragraph


def add_figure(doc, filename):
    path = os.path.join(BASE, filename)
    if not os.path.exists(path):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(6.0))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(cap.add_run(FIG_CAPTIONS.get(filename, filename)), "宋体", 10.5)


def build_docx():
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.54)
    sec.left_margin = sec.right_margin = Cm(3.0)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # blank line
        if not stripped:
            i += 1
            continue

        # table: consecutive lines starting with '|'
        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            # drop the separator row (|---|...)
            rows = [ln for ln in table_lines if not re.match(r"^\|[\s\-|:]+\|$", ln)]
            if not rows:
                continue
            header = [c.strip() for c in rows[0].strip("|").split("|")]
            ncols = len(header)
            table = doc.add_table(rows=0, cols=ncols)
            table.style = "Table Grid"
            for r_idx, row in enumerate(rows):
                cells = [c.strip() for c in row.strip("|").split("|")]
                # pad/trim to ncols
                cells = (cells + [""] * ncols)[:ncols]
                tr_cells = table.add_row().cells
                for c_idx, cell_text in enumerate(cells):
                    cell_obj = tr_cells[c_idx]
                    cell_obj.paragraphs[0].text = ""
                    add_inline(cell_obj.paragraphs[0], cell_text, size=10.5)
                    if r_idx == 0:
                        for run in cell_obj.paragraphs[0].runs:
                            run.bold = True
            doc.add_paragraph()
            continue

        # headings
        if stripped.startswith("#### "):
            p = doc.add_paragraph()
            set_font(p.add_run(stripped[5:]), "黑体", 12, bold=True)
        elif stripped.startswith("### "):
            p = doc.add_paragraph()
            set_font(p.add_run(stripped[4:]), "黑体", 12, bold=True)
        elif stripped.startswith("## "):
            p = doc.add_paragraph()
            set_font(p.add_run(stripped[3:]), "黑体", 14, bold=True)
        elif stripped.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(stripped[2:]), "黑体", 16, bold=True)
        # bullet list
        elif stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, stripped[2:], size=12)
        # figure-caption lines (图N ...) — keep as centered caption
        elif re.match(r"^图\d", stripped):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, stripped, size=10.5)
        # regular paragraph (may contain figure references)
        else:
            text = stripped
            figs = FIG_RE.findall(text)
            cleaned = re.sub(r"[（(][^）)]*fig_[^）)]*[）)]", "", text).strip()
            p = doc.add_paragraph()
            add_inline(p, cleaned, size=12)
            for fig in figs:
                add_figure(doc, fig)
        i += 1

    doc.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    build_docx()
