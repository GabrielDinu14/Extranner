import os
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageOps

import ocr
import page_detection
import word_export

PREVIEW_WIDTH = 800
PREVIEW_HEIGHT = 600
LANGUAGES = {"eng": "eng", "rom & eng": "ron+eng", "rom": "ron"}


def cv_to_pil(image):
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


class ScannerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart OCR Scanner")
        self.geometry("1450x760")

        self.image_path = None
        self.pil_image = None
        self.views = {}

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(bar, text="Load photo", command=self.load_photo).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(bar, text="Detect page", command=self.detect_page).pack(side="left", padx=5)
        ctk.CTkButton(bar, text="Extract text", command=self.extract_text).pack(side="left", padx=5)
        ctk.CTkButton(bar, text="Export to Word", command=self.export_word).pack(side="left", padx=5)

        ctk.CTkLabel(bar, text="Language:").pack(side="left", padx=(15, 5))
        self.language_menu = ctk.CTkOptionMenu(bar, values=list(LANGUAGES), width=100)
        self.language_menu.pack(side="left")

        self.binarize_checkbox = ctk.CTkCheckBox(bar, text="Otsu binarization")
        self.binarize_checkbox.pack(side="left", padx=15)

        ctk.CTkLabel(bar, text="View:").pack(side="left", padx=(15, 5))
        self.view_menu = ctk.CTkOptionMenu(bar, values=["-"], command=self.change_view, width=150)
        self.view_menu.pack(side="left")

        self.info_label = ctk.CTkLabel(self, text="No image loaded", anchor="w")
        self.info_label.pack(fill="x", padx=15)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(expand=True, fill="both", padx=10, pady=10)

        self.image_label = ctk.CTkLabel(main, text="The preview will appear here")
        self.image_label.pack(side="left", expand=True, fill="both")

        self.text_box = ctk.CTkTextbox(main, width=400)
        self.text_box.pack(side="right", fill="y", padx=(10, 0))

    def load_photo(self):
        path = filedialog.askopenfilename(
            title="Choose the document photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )

        if not path:
            return

        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
        except Exception as error:
            self.set_info(f"Error opening file: {error}")
            return

        self.image_path = path
        self.pil_image = image
        self.views = {"Original": image}
        self.update_view_menu("Original")
        self.text_box.delete("1.0", "end")
        width, height = image.size
        self.set_info(f"{os.path.basename(path)}  —  {width} × {height} px")

    def detect_page(self):
        if self.pil_image is None:
            self.set_info("Load a photo first.")
            return

        result = page_detection.process_image(pil_to_cv(self.pil_image))

        self.views["Grayscale"] = cv_to_pil(result["gray"])
        self.views["Gaussian Blur"] = cv_to_pil(result["blurred"])
        self.views["Canny edges"] = cv_to_pil(result["edges"])
        self.views["Contours"] = cv_to_pil(result["drawn_contours"])
        self.update_view_menu("Contours")

        count = len(result["candidates"])
        if count == 0:
            self.set_info("No page contour found (try a darker background or more contrast).")
        else:
            self.set_info(f"Candidate contours found: {count} (green = most likely page)")

    def extract_text(self):
        if self.pil_image is None:
            self.set_info("Load a photo first.")
            return

        image = pil_to_cv(self.pil_image)

        if self.binarize_checkbox.get():
            image = ocr.binarize_otsu(image)
            self.views["Binarized (Otsu)"] = cv_to_pil(image)
            self.update_view_menu("Binarized (Otsu)")

        lang = self.language_menu.get()
        self.set_info(f"Running OCR ({lang})...")
        self.update()

        try:
            text = ocr.extract_text(image, lang)
        except Exception as error:
            self.set_info(f"OCR error: {error}")
            return

        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)
        self.set_info(f"OCR finished: {len(text.split())} words recognized.")

    def export_word(self):
        text = self.text_box.get("1.0", "end-1c")

        if not text.strip():
            self.set_info("There is no text to export. Run 'Extract text' first.")
            return

        name = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "text"
        path = filedialog.asksaveasfilename(
            title="Save as Word document",
            defaultextension=".docx",
            initialfile=f"{name}.docx",
            filetypes=[("Word document", "*.docx")],
        )

        if not path:
            return

        try:
            word_export.save_to_word(text, path)
        except PermissionError:
            self.set_info("Cannot save: the file is open in another program (close it in Word and try again).")
            return
        except Exception as error:
            self.set_info(f"Export error: {error}")
            return

        self.set_info(f"Text saved to: {path}")

    def update_view_menu(self, selected):
        self.view_menu.configure(values=list(self.views))
        self.view_menu.set(selected)
        self.change_view(selected)

    def change_view(self, name):
        if name in self.views:
            self.show_preview(self.views[name])

    def show_preview(self, image):
        width, height = image.size
        factor = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height, 1.0)
        size = (int(width * factor), int(height * factor))

        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=size)
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image

    def set_info(self, text):
        self.info_label.configure(text=text)
