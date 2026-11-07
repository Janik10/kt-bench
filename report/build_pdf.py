"""Render report/report.md to report/report.pdf (Markdown -> HTML -> headless Chrome or Edge).

    pip install -e ".[report]"
    python report/build_pdf.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CSS = """
@page { size: Letter; margin: 0.75in 0.8in; }
body { font-family: "Georgia", "Times New Roman", serif; font-size: 10pt; line-height: 1.36; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }
.byline { text-align: center; color: #444; margin-bottom: 14pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; margin: 6pt auto 10pt; font-size: 8.8pt; }
th, td { border-top: 1px solid #bbb; border-bottom: 1px solid #bbb; padding: 2.5pt 6pt; text-align: left; }
th { background: #f2f1ec; }
img { display: block; max-width: 70%; margin: 4pt auto; }
img[src*="calibration"] { max-width: 40%; }
.refs p { font-size: 8.6pt; line-height: 1.25; margin: 2pt 0; text-align: left; }
code { font-size: 9pt; }
"""


def front_matter(text: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    return {k.strip(): v.strip().strip('"') for k, v in meta.items()}, text[m.end():]


def browser() -> str:
    candidates = [
        os.environ.get("CHROME", ""),
        shutil.which("chrome") or "", shutil.which("chromium") or "", shutil.which("google-chrome") or "",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit("no Chrome/Edge found; set CHROME=/path/to/browser")


def main() -> None:
    meta, body = front_matter((HERE / "report.md").read_text(encoding="utf-8"))
    html_body = markdown.markdown(body, extensions=["tables"])
    html_body = html_body.replace("<h2>References</h2>", "<h2>References</h2><div class='refs'>") + "</div>"
    byline = f"{meta.get('author', '')} · {meta.get('date', '')}"
    head = f"<h1>{meta.get('title', '')}</h1><div class='byline'>{byline}</div>"
    html = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
            f"<body>{head}{html_body}</body></html>")
    html_path = HERE / "report.html"
    html_path.write_text(html, encoding="utf-8")
    pdf = HERE / "report.pdf"
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True)
    print(pdf)


if __name__ == "__main__":
    main()
