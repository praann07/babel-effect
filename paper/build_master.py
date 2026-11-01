"""Build the master record: paper/master_record.md and paper/The_Babel_Effect_MASTER_RECORD.pdf.

The record joins hand-written chapters (paper/master/*.md), documents copied word for word
(the paper, docs/*.md, GOAL.md), and lists generated from the repository (layout, run folders,
git history). Edit a source, rerun this script, get a new PDF.

Usage (from the repo root):   .venv/Scripts/python.exe paper/build_master.py
Needs: pandoc on PATH and Microsoft Edge (used headless to print the PDF).
Options: --no-pdf   write the Markdown only.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MASTER = PAPER / "master"
OUT_MD = PAPER / "master_record.md"
OUT_PDF = PAPER / "The_Babel_Effect_MASTER_RECORD.pdf"
EDGE = [Path(p) for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]
TITLE = "The Babel Effect: master record"
SUB = "Goal, story, problems faced, paper, pre-registrations, logs and paths: everything in one file"


def read(rel: str | Path) -> str:
    p = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
    return p.read_text(encoding="utf8").replace("\r\n", "\n").strip("\n") + "\n"


def shift(text: str, by: int) -> str:
    """Push every Markdown heading down `by` levels (outside code fences); six levels at most."""
    out, fence = [], False
    for line in text.split("\n"):
        if re.match(r"^\s*(```|~~~)", line):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,6})(\s+.*)$", line)
        out.append(("#" * min(6, len(m.group(1)) + by) + m.group(2)) if m else line)
    return "\n".join(out)


def source_note(rel: str) -> str:
    return f"*Copied word for word from `{rel}` when this record was built. To change it, edit that file and rebuild.*\n"


def captions() -> dict[str, str]:
    """Figure captions are kept in paper/build_tex.py (FIGS); read them from the text without running that script."""
    src = (PAPER / "build_tex.py").read_text(encoding="utf8")
    caps = {}
    for m in re.finditer(r'\("(fig\d_[a-z0-9_]+)\.(?:pdf|png)",\s*[0-9.]+,\s*"((?:[^"\\]|\\.)*)"\)', src):
        c = m.group(2).replace("\\\\%", "%").replace("\\%", "%").replace("Figure~\\\\ref{fig:sw1}", "Figure 5").replace("Figure~\\ref{fig:sw1}", "Figure 5")
        caps[m.group(1)] = c
    return caps


def figures_section() -> str:
    caps, rows = captions(), []
    for i, p in enumerate(sorted((PAPER / "figures").glob("fig*.png")), 1):
        n = re.match(r"fig(\d)", p.name).group(1)
        rows.append(f"![Figure {n}. {caps.get(p.stem, p.stem)}](paper/figures/{p.name})\n")
    return "## Figures of the paper\n\nThe figure files are in `paper/figures/` (PNG and PDF). Captions are the ones used in the LaTeX build (`paper/build_tex.py`).\n\n" + "\n".join(rows)


def run(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf8", errors="replace", check=kw.pop("check", True), **kw).stdout


def layout_block() -> str:
    lines = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in {"models", "__pycache__"}):
        n = sum(1 for f in d.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
        lines.append(f"{d.name + '/':<14}{n:>6} files")
    top = sorted(p.name for p in ROOT.iterdir() if p.is_file() and not p.name.startswith("."))
    return "```text\n" + "\n".join(lines) + "\n\nFiles at the top level:\n" + "\n".join(f"  {t}" for t in top) + "\n```\n"


def runs_block() -> str:
    lines = [f"{'folder':<62}{'files':>6}{'KB':>9}"]
    for d in sorted((ROOT / "analysis").iterdir()):
        if not d.is_dir():
            continue
        fs = [f for f in d.rglob("*") if f.is_file()]
        lines.append(f"{d.name:<62}{len(fs):>6}{sum(f.stat().st_size for f in fs) // 1024:>9}")
    return "```text\n" + "\n".join(lines) + "\n```\n"


def git_block() -> str:
    log = run(["git", "log", "--date=short", "--format=%h  %ad  %s"], check=False).strip()
    n = len(log.splitlines())
    return f"{n} commits at the time of building. Newest first.\n\n```text\n{log}\n```\n"


def assemble() -> str:
    today = dt.date.today().isoformat()
    paper = shift(read("paper/paper_v2_draft.md"), 1)
    parts = [
        f"---\ntitle: \"{TITLE}\"\nsubtitle: \"{SUB}\"\ndate: \"Built {today} from the repository at paper/build_master.py\"\n---\n",
        read(MASTER / "00_front.md"),
        read(MASTER / "01_goal_and_rules.md"),
        read(MASTER / "02_story.md"),
        read(MASTER / "03_problems.md"),
        read(MASTER / "04_words_and_mistakes.md"),
        "# Part 5. The paper\n\n" + source_note("paper/paper_v2_draft.md") + "\n" + paper + "\n" + figures_section()
        + "\n## Appendix: full overlap tables\n\n" + source_note("paper/tables/overlap_full.md") + "\n" + shift(read("paper/tables/overlap_full.md"), 2),
        read(MASTER / "06_results_glance.md"),
        read(MASTER / "07_open.md"),
        "# Part 8. Pre-registrations\n\n" + source_note("docs/HYPOTHESIS.md") + "\n" + shift(read("docs/HYPOTHESIS.md"), 1),
        read(MASTER / "05_paths.md")
        + "\n## 9.7 Repository layout (generated)\n\n" + layout_block()
        + "\n## 9.8 Run folders in `analysis/` (generated)\n\nWrite-once; one folder per run. Names carry the date, the test and the model.\n\n" + runs_block()
        + "\n## 9.9 Git history (generated)\n\n" + git_block(),
        "# Part 10. Logs and plans\n\nEach chapter below is copied unedited. Run codes (A1 to A30) are explained in Part 8.\n",
    ]
    for num, title, rel in [
        ("10.1", "Master goal and plan file", "GOAL.md"),
        ("10.2", "Decision log", "docs/LOG.md"),
        ("10.3", "Experiment log", "docs/EXPERIMENT_LOG.md"),
        ("10.4", "Next steps", "docs/NEXT.md"),
        ("10.5", "Q1 roadmap", "docs/ROADMAP_Q1.md"),
        ("10.6", "Outside review of 2026-10-23", "docs/REVIEW_2026-10-23.md"),
        ("10.7", "Draft of the control analysis (A12a)", "docs/A12a_DRAFT.md"),
    ]:
        parts.append(f"## {num} {title}\n\n{source_note(rel)}\n{shift(read(rel), 2)}")
    return "\n\n".join(p.strip("\n") + "\n" for p in parts)


CSS = """
@page { size: A4; margin: 18mm 15mm 18mm 15mm; @bottom-center { content: counter(page); font: 9pt Segoe UI, sans-serif; color: #555; } }
html { font: 10.2pt/1.45 "Segoe UI", Calibri, "Nirmala UI", sans-serif; color: #1b1b1b; }
body { max-width: none; margin: 0; padding: 0; }
header#title-block-header { text-align: center; margin: 30mm 0 20mm; }
h1.title { font-size: 28pt; margin-bottom: 4pt; } p.subtitle { font-size: 13pt; color: #444; } p.date { color: #666; font-size: 9pt; }
h1 { font-size: 20pt; border-bottom: 2px solid #1f4e79; padding-bottom: 3pt; margin-top: 0; page-break-before: always; color: #1f4e79; }
header#title-block-header h1 { page-break-before: avoid; border: 0; }
h2 { font-size: 14pt; color: #1f4e79; margin-top: 16pt; } h3 { font-size: 11.5pt; } h4, h5, h6 { font-size: 10.5pt; }
h1, h2, h3, h4 { break-after: avoid; }
nav#TOC { page-break-after: always; font-size: 9.2pt; } nav#TOC ul { padding-left: 14pt; margin: 1pt 0; list-style: none; } nav#TOC a { text-decoration: none; color: #1b1b1b; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.6pt; break-inside: auto; }
th, td { border: 1px solid #b9c3cf; padding: 3pt 5pt; vertical-align: top; overflow-wrap: anywhere; text-align: left; } th { background: #e8eef5; }
tr { break-inside: avoid; }
code { font: 8.6pt Consolas, "Cascadia Mono", monospace; background: #f1f3f5; padding: 0 2pt; overflow-wrap: anywhere; }
pre { background: #f6f7f9; border: 1px solid #dde1e6; padding: 6pt; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 8.2pt; break-inside: auto; }
pre code { background: none; padding: 0; font-size: 8.2pt; }
blockquote { border-left: 3px solid #9fb3c8; margin: 6pt 0 6pt 4pt; padding: 0 10pt; color: #333; }
img { max-width: 100%; height: auto; } figure { margin: 8pt 0; break-inside: avoid; } figcaption { font-size: 8.8pt; color: #333; }
a { color: #1f4e79; word-break: break-word; }
"""


def find_edge() -> Path:
    for p in EDGE:
        if p.exists():
            return p
    raise SystemExit("Microsoft Edge not found; run with --no-pdf or install Edge.")


def build_pdf(md: Path, pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        css, html = tmp / "master.css", tmp / "master.html"
        css.write_text(CSS, encoding="utf8")
        subprocess.run(
            ["pandoc", str(md), "-f", "markdown-smart-tex_math_dollars-raw_html-fancy_lists-example_lists-superscript-subscript", "-t", "html5", "--standalone", "--embed-resources",
             "--toc", "--toc-depth=2", "--css", str(css), "--metadata", f"pagetitle={TITLE}", "--resource-path", str(ROOT), "-o", str(html)],
            cwd=ROOT, check=True)
        subprocess.run([str(find_edge()), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, timeout=900, capture_output=True)


def main() -> int:
    md = assemble()
    OUT_MD.write_text(md, encoding="utf8")
    print(f"wrote {OUT_MD.relative_to(ROOT)}: {len(md) // 1024} KB, {md.count(chr(10))} lines")
    if "--no-pdf" in sys.argv:
        return 0
    build_pdf(OUT_MD, OUT_PDF)
    print(f"wrote {OUT_PDF.relative_to(ROOT)}: {OUT_PDF.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
