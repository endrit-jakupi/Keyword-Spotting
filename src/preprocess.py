import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import random

# Optional: make processing reproducible
np.random.seed(42)
random.seed(42)

def load_polygons(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    polygons = {}

    # Handle both <polygon> and <path> elements
    for elem in root.findall(".//{http://www.w3.org/2000/svg}*"):
        tag = elem.tag.split('}')[-1]
        if tag not in ["polygon", "path", "polyline"]:
            continue

        word_id = elem.get("id") or f"poly_{len(polygons)}"

        if tag in ["polygon", "polyline"] and "points" in elem.attrib:
            points_str = re.sub(r"\s+", " ", elem.attrib["points"].strip())
            pts = [tuple(map(float, p.split(","))) for p in points_str.split()]
        elif tag == "path" and "d" in elem.attrib:
            d = elem.attrib["d"]
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", d)
            pts = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords) - 1, 2)]
        else:
            continue

        if len(pts) > 2:
            polygons[word_id] = np.array(pts, np.int32)

    return polygons


def crop_word(image_path, polygon):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    result = cv2.bitwise_and(image, mask)
    x, y, w, h = cv2.boundingRect(polygon)
    crop = result[y:y + h, x:x + w]
    return crop


def sauvola_binarization(image, window_size=25, k=0.5):
    image = image.astype(np.float32)
    mean = cv2.blur(image, (window_size, window_size))
    mean_sq = cv2.blur(image ** 2, (window_size, window_size))
    std = np.sqrt(mean_sq - mean ** 2)
    R = np.max(std)
    thresh = mean * (1 + k * ((std / R) - 1))
    binary = (image > thresh).astype(np.uint8) * 255
    return binary


def normalize_image(image, size=(100, 100)):
    img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img



def preprocess_word(image_path, polygon):
    img = crop_word(image_path, polygon)
    img = sauvola_binarization(img)
    img = normalize_image(img)
    return img


def load_metadata(dataset_dir):
    paths = {
        "train": Path(dataset_dir) / "train.tsv",
        "validation": Path(dataset_dir) / "validation.tsv",
        "transcriptions": Path(dataset_dir) / "transcription.tsv",
        "keywords": Path(dataset_dir) / "keywords.tsv",
    }
    metadata = {}
    for key, path in paths.items():
        if path.exists():
            metadata[key] = pd.read_csv(path, sep="\t")
        else:
            print(f"[Warning] Missing file: {path}")
            metadata[key] = None
    return metadata
