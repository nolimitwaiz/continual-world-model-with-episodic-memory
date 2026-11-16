#!/usr/bin/env python3
"""Build the two-page research proposal from its Markdown source.

Optional dependencies: reportlab, pypdf. The overview PNG can be regenerated
with scripts/render_figures.cjs. No training, network calls, or data downloads.
"""
from pathlib import Path
from html import escape
import re
from urllib.parse import urljoin

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/research-brief.md"
OUTPUT = ROOT / "output/pdf/Remembering-Events-Research-Proposal.pdf"
WEB_DOCS = "https://github.com/nolimitwaiz/continual-world-model-with-episodic-memory/blob/main/docs/"


def inline(text):
    def link(match):
        label, target = match.groups()
        url = target if target.startswith("https://") else urljoin(WEB_DOCS, target)
        return f'<link href="{escape(url, quote=True)}" color="#087E78">{escape(label)}</link>'
    parts = []
    last = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        parts.extend([escape(text[last:match.start()]), link(match)])
        last = match.end()
    parts.append(escape(text[last:]))
    return "".join(parts)


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#DCE4EA"))
    canvas.line(46, 40, 566, 40)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526579"))
    canvas.drawString(46, 27, "Research discussion proposal | Development findings, not independent confirmation")
    canvas.drawRightString(566, 27, str(document.page))
    canvas.restoreState()


def main():
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ResearchBody", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=10.1, leading=14.2, spaceAfter=8.0,
                          textColor=colors.HexColor("#172C43"))
    heading = ParagraphStyle("ResearchHeading", parent=body, fontName="Helvetica-Bold",
                             fontSize=13, leading=17, spaceBefore=10, spaceAfter=6, keepWithNext=True)
    title = ParagraphStyle("ResearchTitle", parent=heading, fontSize=22, leading=26,
                           spaceBefore=0, spaceAfter=10)
    small = ParagraphStyle("ResearchSmall", parent=body, fontSize=8.8, leading=12, spaceAfter=5)
    story = []
    for block in re.split(r"\n\s*\n", SOURCE.read_text().strip()):
        text = " ".join(line.strip() for line in block.splitlines())
        if text == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif text.startswith("# "):
            story.append(Paragraph(inline(text[2:]), title))
        elif text.startswith("## "):
            story.append(Paragraph(inline(text[3:]), heading))
        elif text.startswith("!["):
            path = ROOT / "docs/figures/overview.png"
            width, height = ImageReader(str(path)).getSize()
            image = Image(str(path), width=520, height=520 * height / width)
            story.append(KeepTogether([Spacer(1, 3), image, Spacer(1, 7)]))
        elif text.startswith("- "):
            for line in block.splitlines():
                if line.startswith("- "):
                    story.append(Paragraph(inline(line[2:]), small))
        elif text.startswith("Waiz Khan"):
            story.append(Paragraph(inline(text), small))
        else:
            story.append(Paragraph(inline(text), body))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(612, 792), leftMargin=46,
                           rightMargin=46, topMargin=40, bottomMargin=52,
                           title="What Should a Model Remember About an Event?",
                           author="Waiz Khan", subject="Research discussion proposal")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(OUTPUT)
    if len(reader.pages) != 2:
        raise RuntimeError(f"Expected a two-page proposal; got {len(reader.pages)}")
    text = "\n".join(page.extract_text() for page in reader.pages)
    for required in ("73.55%", "56.29%", "69.77%", "71.53%", "The Next Experiment"):
        if required not in text:
            raise RuntimeError(f"Missing proposal content: {required}")
    print(f"Created a verified two-page proposal: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
