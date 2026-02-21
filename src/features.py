import cv2
import numpy as np

def extract_features(image):

    if image.dtype != np.uint8:
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    else:
        img = image

    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    features = hog.compute(img).flatten()
    features = features / (np.linalg.norm(features) + 1e-8)

    return features
