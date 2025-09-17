"""Markdown to PDF conversion with professional formatting."""

from __future__ import annotations

from html import escape
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from markdown import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=10,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=4,
    )
    numbered = ParagraphStyle(
        "Numbered",
        parent=body,
        leftIndent=22,
        bulletIndent=6,
        spaceAfter=4,
    )
    code = ParagraphStyle(
        "Code",
        parent=body,
        fontName="Courier",
        backColor=colors.whitesmoke,
        leftIndent=6,
        rightIndent=6,
        leading=12,
    )
    blockquote = ParagraphStyle(
        "Blockquote",
        parent=body,
        leftIndent=18,
        textColor=colors.HexColor("#555555"),
        italic=True,
    )
    heading1 = ParagraphStyle(
        "Heading1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1f3d7a"),
        spaceBefore=16,
        spaceAfter=12,
    )
    heading2 = ParagraphStyle(
        "Heading2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1f3d7a"),
        spaceBefore=14,
        spaceAfter=8,
    )
    heading3 = ParagraphStyle(
        "Heading3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1f3d7a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    return {
        "body": body,
        "bullet": bullet,
        "numbered": numbered,
        "code": code,
        "blockquote": blockquote,
        "h1": heading1,
        "h2": heading2,
        "h3": heading3,
    }


def _emit_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br />"), style)


def _emit_list(element, numbered: bool, style: ParagraphStyle) -> list:
    flowables = []
    for idx, item in enumerate(element.find_all("li", recursive=False), start=1):
        content = " ".join(s.strip() for s in item.stripped_strings)
        if not content:
            continue
        if numbered:
            text = f"<bullet>{idx}.</bullet>{escape(content)}"
        else:
            text = f"<bullet>&bull;</bullet>{escape(content)}"
        flowables.append(Paragraph(text, style))
    if flowables:
        flowables.append(Spacer(1, 0.08 * inch))
    return flowables


def _emit_table(element) -> list:
    rows: list[list[str]] = []
    header_rows = 0
    for row in element.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        text_row = [escape(cell.get_text(strip=True)) for cell in cells]
        rows.append(text_row)
        if header_rows == 0 and any(cell.name == "th" for cell in cells):
            header_rows = 1
    if not rows:
        return []
    table = Table(rows, hAlign="LEFT")
    table_style = TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]
    )
    if header_rows:
        table_style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3d7a"))
        table_style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke)
    table.setStyle(table_style)
    return [table, Spacer(1, 0.12 * inch)]


def _emit_preformatted(element, style: ParagraphStyle) -> Paragraph:
    text = element.get_text("\n")
    escaped = escape(text).replace("\n", "<br />")
    return Paragraph(f"<font face='Courier'>{escaped}</font>", style)


def _header_footer(title: str):
    def draw(canvas_obj: canvas.Canvas, doc) -> None:  # type: ignore[override]
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(colors.HexColor("#666666"))
        if title:
            canvas_obj.drawString(doc.leftMargin, doc.pagesize[1] - doc.topMargin + 20, title)
        canvas_obj.drawRightString(
            doc.pagesize[0] - doc.rightMargin,
            doc.bottomMargin - 20,
            f"Page {doc.page}",
        )
        canvas_obj.restoreState()

    return draw


def markdown_to_pdf(md_text: str, out_path: Path) -> None:
    """Render markdown content to a styled PDF document."""

    html = markdown(md_text, extensions=["extra", "sane_lists", "tables"])
    soup = BeautifulSoup(html, "html.parser")
    container = soup.body if soup.body else soup
    styles = _build_styles()

    story: list = []
    title = ""

    for element in container.children:
        if isinstance(element, NavigableString):
            continue
        name = element.name
        if name in {"h1", "h2", "h3"}:
            text = element.get_text(strip=True)
            if name == "h1" and not title:
                title = text
            story.append(_emit_paragraph(text, styles[name]))
        elif name == "p":
            text = element.get_text(strip=True)
            if text:
                story.append(_emit_paragraph(text, styles["body"]))
        elif name in {"ul", "ol"}:
            numbered = name == "ol"
            style = styles["numbered" if numbered else "bullet"]
            story.extend(_emit_list(element, numbered, style))
        elif name == "table":
            story.extend(_emit_table(element))
        elif name in {"pre", "code"}:
            story.append(_emit_preformatted(element, styles["code"]))
        elif name == "blockquote":
            text = element.get_text(strip=True)
            if text:
                story.append(_emit_paragraph(text, styles["blockquote"]))
        else:
            text = element.get_text(strip=True)
            if text:
                story.append(_emit_paragraph(text, styles["body"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title=title or "BDSSim Report",
    )
    header_footer = _header_footer(title or "BDSSim Report")
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


__all__ = ["markdown_to_pdf"]
