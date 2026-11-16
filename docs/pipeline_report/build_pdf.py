#!/usr/bin/env python3
"""
build_pdf.py - turn pipeline_report.md into a typeset, self-contained HTML and a PDF.

* Equations ($...$ and $$...$$, with \\tag{n}) are typeset with matplotlib's
  mathtext (Computer Modern), so no LaTeX installation is needed.
* Figures are embedded, so pipeline_report.html is a single file.
* The PDF is printed by Microsoft Edge (or Chrome) in headless mode through WSL interop.

Run from the project folder:  python3 docs/pipeline_report/build_pdf.py
"""
import base64
import html
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

import markdown
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["svg.fonttype"] = "path"
from matplotlib import mathtext  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "pipeline_report.md")
OUT_HTML = os.path.join(HERE, "pipeline_report.html")
OUT_PDF = os.path.join(HERE, "pipeline_report.pdf")


def tex_to_img(tex, size, display=False):
    buf = io.BytesIO()
    depth = mathtext.math_to_image(f"${tex}$", buf, prop=FontProperties(size=size), dpi=72, format="svg")
    svg = buf.getvalue()
    m = re.search(rb'height="([\d.]+)pt"', svg)
    h = float(m.group(1)) if m else size
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    alt = html.escape(tex, quote=True)
    if display:
        return f'<img class="dmath" src="{uri}" alt="{alt}" style="height:{h:.2f}pt">'
    return f'<img class="imath" src="{uri}" alt="{alt}" style="height:{h:.2f}pt;vertical-align:-{float(depth):.2f}pt">'


def build_html():
    src = open(MD, encoding="utf-8").read()
    store = {}
    errors = []

    def keep(fragment):
        key = f"MATHTOKEN{len(store):05d}Z"
        store[key] = fragment
        return key

    def display(m):
        tex = m.group(1).strip()
        tag = re.search(r"\\tag\{([^}]*)\}", tex)
        tex = re.sub(r"\\tag\{[^}]*\}", "", tex).strip()
        try:
            img = tex_to_img(tex, 12.5, display=True)
        except Exception as e:  # report and keep going
            errors.append((tex, str(e).splitlines()[0]))
            img = f"<code>{html.escape(tex)}</code>"
        no = f'<span class="eqno">({tag.group(1)})</span>' if tag else ""
        return "\n\n" + keep(f'<div class="eq">{img}{no}</div>') + "\n\n"

    def inline(m):
        tex = m.group(1)
        try:
            return keep(tex_to_img(tex, 11.2))
        except Exception as e:
            errors.append((tex, str(e).splitlines()[0]))
            return keep(f"<code>{html.escape(tex)}</code>")

    # protect code first so $ inside code is never treated as math
    codes = {}

    def keep_code(m):
        key = f"CODETOKEN{len(codes):05d}Z"
        codes[key] = m.group(0)
        return key

    src = re.sub(r"```.*?```", keep_code, src, flags=re.S)
    src = re.sub(r"`[^`\n]+`", keep_code, src)
    src = re.sub(r"\$\$(.+?)\$\$", display, src, flags=re.S)
    src = re.sub(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$", inline, src)
    for k, v in codes.items():
        src = src.replace(k, v)

    body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])
    for k, v in store.items():
        body = body.replace(f"<p>{k}</p>", v).replace(k, v)

    # embed figures and turn "image + **Figure n.** caption" into <figure>
    def embed(m):
        path = os.path.join(HERE, m.group(2))
        data = base64.b64encode(open(path, "rb").read()).decode()
        return f'<img alt="{m.group(1)}" src="data:image/png;base64,{data}"'

    body = re.sub(r'<img alt="([^"]*)" src="([^"]+\.png)"', embed, body)
    body = re.sub(r"<p>(<img alt=\"Figure[^>]*>)</p>\s*<p><strong>(Figure \d+\.)</strong>(.*?)</p>",
                  r'<figure>\1<figcaption><strong>\2</strong>\3</figcaption></figure>', body, flags=re.S)
    body = re.sub(r"<p><strong>(Table \d+\.)</strong>(.*?)</p>", r'<p class="tcap"><strong>\1</strong>\2</p>', body, flags=re.S)
    # title block: first h1 + italic line under it
    body = re.sub(r"<h1>(.*?)</h1>\s*<p><em>(.*?)</em></p>", r'<header class="title"><h1>\1</h1><p class="meta">\2</p></header>', body, count=1, flags=re.S)
    body = re.sub(r'(<h2>Abstract</h2>)(.*?)(<hr />)', r'<section class="abstract">\1\2</section>', body, count=1, flags=re.S)
    return body, errors


CSS = r"""
@page { size: A4; margin: 19mm 18mm 20mm 18mm;
  @bottom-center { content: counter(page); font: 9pt Cambria, Georgia, serif; color: #555; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: Cambria, "Cambria Math", Georgia, "Times New Roman", serif; font-size: 10.6pt; line-height: 1.45;
  color: #111; max-width: 174mm; margin: 0 auto; text-align: justify; hyphens: auto; }
header.title { text-align: center; margin: 4mm 0 5mm; }
header.title h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 3mm; font-weight: 700; text-align: center; }
header.title .meta { font-style: italic; color: #444; font-size: 10pt; margin: 0; text-align: center; }
section.abstract { border-top: 0.6pt solid #222; border-bottom: 0.6pt solid #222; padding: 2mm 6mm 3mm; margin: 0 0 6mm; font-size: 10pt; }
section.abstract h2 { font-size: 10.5pt; text-transform: uppercase; letter-spacing: 0.08em; text-align: center; margin: 2mm 0; }
h2 { font-size: 13.5pt; margin: 7mm 0 2.5mm; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 2mm; break-after: avoid; }
hr { border: 0; border-top: 0.4pt solid #bbb; margin: 5mm 0; }
p { margin: 0 0 2.4mm; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 2.4mm; padding-left: 6mm; } li { margin: 0.6mm 0; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 9pt; background: #f2f2f0; padding: 0 2px; border-radius: 2px; }
pre { background: #f4f4f2; border: 0.4pt solid #ddd; padding: 2mm 3mm; font-size: 8.6pt; overflow: hidden; white-space: pre-wrap; text-align: left; hyphens: none; break-inside: avoid; }
pre code { background: none; padding: 0; }
.eq { display: flex; align-items: center; justify-content: center; position: relative; margin: 3mm 0 3.4mm; break-inside: avoid; }
.eq .eqno { position: absolute; right: 0; font-size: 10.4pt; }
img.dmath { max-width: 88%; }
img.imath { display: inline; }
figure { margin: 4mm 0 5mm; break-inside: avoid; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 105mm; width: auto; height: auto; }
figcaption { font-size: 9.2pt; line-height: 1.38; text-align: justify; margin-top: 1.6mm; color: #222; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 2mm 0 4mm; break-inside: avoid; }
thead th { border-top: 0.9pt solid #111; border-bottom: 0.6pt solid #111; padding: 1.2mm 1.6mm; text-align: left; font-weight: 700; }
tbody td { padding: 1mm 1.6mm; border-bottom: 0.3pt solid #ddd; text-align: left; vertical-align: top; }
tbody tr:last-child td { border-bottom: 0.9pt solid #111; }
p.tcap { font-size: 9.2pt; margin-top: 4mm; break-after: avoid; }
strong { font-weight: 700; }
"""


def find_browser():
    cands = ["/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
             "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
             "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"]
    for c in cands:
        if os.path.exists(c):
            return c
    for name in ["chromium", "chromium-browser", "google-chrome"]:
        p = shutil.which(name)
        if p:
            return p
    return None


def print_pdf(html_path, pdf_path):
    browser = find_browser()
    if not browser:
        print("No Edge/Chrome found: open pipeline_report.html and print it to PDF manually.")
        return False
    windows = browser.endswith(".exe")
    if windows:
        win_tmp = subprocess.run(["cmd.exe", "/c", "echo %TEMP%"], capture_output=True, text=True, cwd="/mnt/c").stdout.strip()
        tmp_dir = subprocess.run(["wslpath", "-u", win_tmp], capture_output=True, text=True).stdout.strip()
        work = tempfile.mkdtemp(prefix="pipeline_report_", dir=tmp_dir)
    else:
        work = tempfile.mkdtemp(prefix="pipeline_report_")
    h, p = os.path.join(work, "report.html"), os.path.join(work, "report.pdf")
    shutil.copy(html_path, h)
    conv = (lambda x: subprocess.run(["wslpath", "-w", x], capture_output=True, text=True).stdout.strip()) if windows else (lambda x: x)
    url = "file:///" + conv(h).replace("\\", "/")
    args = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
            "--virtual-time-budget=8000", f"--print-to-pdf={conv(p)}", url]
    subprocess.run(args, capture_output=True, text=True, timeout=180, cwd="/mnt/c" if windows else None)
    if not os.path.exists(p):
        print("The browser did not write a PDF.")
        return False
    shutil.copy(p, pdf_path)
    shutil.rmtree(work, ignore_errors=True)
    return True


def main():
    body, errors = build_html()
    if errors:
        print("Equations that could not be typeset:")
        for tex, err in errors:
            print("  ", tex, "->", err)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Cable Segmentation and Robot Pickup-Point Selection</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {OUT_HTML} ({os.path.getsize(OUT_HTML) / 1e6:.1f} MB)")
    if "--html-only" not in sys.argv and print_pdf(OUT_HTML, OUT_PDF):
        print(f"Wrote {OUT_PDF} ({os.path.getsize(OUT_PDF) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
