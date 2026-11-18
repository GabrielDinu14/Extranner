import argparse
import json
import os
import sys
import time
import unicodedata

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ocr
import page_detection

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "ground_truth")
DATA = os.path.join(HERE, "..", "test_data")


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalize(text):
    return " ".join(text.split())


def strip_diacritics(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def cer(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp)

    return levenshtein(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    ref, hyp = normalize(ref).split(), normalize(hyp).split()

    return levenshtein(ref, hyp) / max(len(ref), 1)


def jaccard(poly_a, poly_b, shape):
    mask_a = np.zeros(shape, np.uint8)
    mask_b = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask_a, [np.int32(poly_a)], 1)
    cv2.fillPoly(mask_b, [np.int32(poly_b)], 1)
    union = np.logical_or(mask_a, mask_b).sum()

    return np.logical_and(mask_a, mask_b).sum() / union if union else 0.0


def binarize_adaptive(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


VARIANTS = {"none": lambda img: img, "otsu": ocr.binarize_otsu, "adaptive": binarize_adaptive}


def read_gt(name):
    with open(os.path.join(GT, name + ".txt"), encoding="utf-8") as f:
        return f.read()


def find(name):
    for ext in (".png", ".jpg"):
        path = os.path.join(DATA, name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="eng")
    args = parser.parse_args()
    results = {"lang": args.lang}

    with open(os.path.join(GT, "corners.json")) as f:
        corners = json.load(f)
    det = []
    for name, gt_poly in corners.items():
        image = cv2.imread(find(name))
        start = time.perf_counter()
        r = page_detection.process_image(image)
        elapsed = time.perf_counter() - start
        shape = image.shape[:2]
        if r["candidates"]:
            hull = cv2.convexHull(r["candidates"][0]).reshape(-1, 2) / r["ratio"]
            iou = jaccard(hull, gt_poly, shape)
        else:
            iou = 0.0
        h, w = shape
        full_frame = jaccard([[0, 0], [w, 0], [w, h], [0, h]], gt_poly, shape)
        det.append({"image": name, "found": bool(r["candidates"]), "iou": iou,
                    "iou_full_frame": full_frame, "time_ms": elapsed * 1000})
        print(f"[det] {name:28s} found={bool(r['candidates'])!s:5s} IoU={iou:.3f} "
              f"full-frame={full_frame:.3f} {elapsed * 1000:.0f} ms")
    results["detection"] = det

    rows = []
    for name in ["scanned_page", "ocr1_romanian_diacritics", "ocr2_english_invoice", "ocr3_small_text",
                 "ocr4_uneven_lighting", "ocr5_noise"]:
        image = cv2.imread(find(name))
        ref = read_gt(name)
        for variant, fn in VARIANTS.items():
            start = time.perf_counter()
            hyp = ocr.extract_text(fn(image), args.lang)
            elapsed = time.perf_counter() - start
            row = {"image": name, "variant": variant, "cer": cer(ref, hyp), "wer": wer(ref, hyp),
                   "cer_no_diacritics": cer(strip_diacritics(ref), strip_diacritics(hyp)), "time_s": elapsed}
            rows.append(row)
            print(f"[ocr] {name:28s} {variant:9s} CER={row['cer']:.3f} WER={row['wer']:.3f} "
                  f"CER(no diacr.)={row['cer_no_diacritics']:.3f} {elapsed:.2f}s")
    results["ocr"] = rows

    photos = []
    for name in ["document_photo", "det1_clear", "det2_tilted", "det5_shadow"]:
        image = cv2.imread(find(name))
        ref = read_gt(name)
        raw = ocr.extract_text(image, args.lang)
        src = np.float32(corners[name])
        dst = np.float32([[0, 0], [1240, 0], [1240, 1754], [0, 1754]])
        warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (1240, 1754))
        corrected = ocr.extract_text(warped, args.lang)
        corrected_otsu = ocr.extract_text(ocr.binarize_otsu(warped), args.lang)
        corrected_adaptive = ocr.extract_text(binarize_adaptive(warped), args.lang)
        row = {"image": name,
               "cer_raw": cer(ref, raw), "wer_raw": wer(ref, raw),
               "cer_warp": cer(ref, corrected), "wer_warp": wer(ref, corrected),
               "cer_warp_otsu": cer(ref, corrected_otsu), "wer_warp_otsu": wer(ref, corrected_otsu),
               "cer_warp_adaptive": cer(ref, corrected_adaptive), "wer_warp_adaptive": wer(ref, corrected_adaptive)}
        photos.append(row)
        print(f"[photo] {name:26s} raw CER={row['cer_raw']:.3f}  warp CER={row['cer_warp']:.3f}  "
              f"warp+otsu CER={row['cer_warp_otsu']:.3f}  warp+adaptive CER={row['cer_warp_adaptive']:.3f}")
    results["photos"] = photos

    out = os.path.join(HERE, f"results_{args.lang.replace('+', '_')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("Saved:", out)


if __name__ == "__main__":
    main()
