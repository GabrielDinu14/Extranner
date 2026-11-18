import re

from docx import Document
from docx.shared import Pt


def clean_text(text):
    return re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)


def save_to_word(text, path):
    document = Document()

    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    for line in clean_text(text).strip().split("\n"):
        document.add_paragraph(line)

    document.save(path)
