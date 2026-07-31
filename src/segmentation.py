import cv2
import numpy as np


def segment_by_difference(img, background, threshold=20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, gray_bg)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def segment_by_grabcut(
    img, rect_proportion=(0.05, 0.40, 0.90, 0.55), iterations=5,
):
    h, w = img.shape[:2]
    x, y, rw, rh = rect_proportion
    rect = (int(x * w), int(y * h), int(rw * w), int(rh * h))
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        img, mask, rect, bgd_model, fgd_model,
        iterations, cv2.GC_INIT_WITH_RECT,
    )
    return np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")


def apply_morphology(mask, kernel_size=7):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_largest_contour(mask, min_area=500):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return largest


def compute_centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def make_filled_mask(contour, reference_mask):
    mask = np.zeros_like(reference_mask)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask
