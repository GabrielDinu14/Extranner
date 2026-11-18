import cv2

def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Cannot open image: {path}")

    return image


def resize(image, target_height=800):
    h, w = image.shape[:2]

    if h <= target_height:
        return image, 1.0

    ratio = target_height / h
    resized = cv2.resize(image, (int(w * ratio), target_height), interpolation=cv2.INTER_AREA)

    return resized, ratio


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image, kernel_size=5):
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def detect_edges(blurred_image, low_threshold=75, high_threshold=200):
    return cv2.Canny(blurred_image, low_threshold, high_threshold)


def find_candidate_contours(edges, max_count=5, min_area_percent=10):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = edges.shape[0] * edges.shape[1]
    min_area = image_area * min_area_percent / 100

    candidates = [c for c in contours if cv2.contourArea(cv2.convexHull(c)) >= min_area]
    candidates.sort(key=lambda c: cv2.contourArea(cv2.convexHull(c)), reverse=True)

    return candidates[:max_count]


def draw_contours(image, contours):
    result = image.copy()

    for i, contour in enumerate(contours):
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.drawContours(result, [contour], -1, color, 3)

    return result


def process(image_path):
    return process_image(load_image(image_path))


def process_image(original):
    small, ratio = resize(original)
    gray = to_grayscale(small)
    blurred = apply_gaussian_blur(gray)
    edges = detect_edges(blurred)
    candidates = find_candidate_contours(edges)

    return {
        "image": small,
        "ratio": ratio,
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "candidates": candidates,
        "drawn_contours": draw_contours(small, candidates),
    }
