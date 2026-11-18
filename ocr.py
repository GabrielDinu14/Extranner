import cv2
import pytesseract

# PATH to tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def check_tesseract():
    try:
        version = pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        raise SystemExit("Tesseract was not found. Install it and/or set tesseract_cmd in ocr.py.")

    languages = pytesseract.get_languages(config="")
    print(f"Tesseract {version}, available languages: {', '.join(languages)}")

    return languages


def binarize_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def extract_text(image, lang="eng"):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return pytesseract.image_to_string(image, lang=lang, config="--psm 3")


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
