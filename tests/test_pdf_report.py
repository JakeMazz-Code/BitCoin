from pathlib import Path

from bdssim.reporting.pdf import markdown_to_pdf


def test_markdown_to_pdf(tmp_path: Path) -> None:
    md = "# Title\n\nParagraph text.\n\n- Bullet one\n- Bullet two"
    out_path = tmp_path / "report.pdf"
    markdown_to_pdf(md, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
