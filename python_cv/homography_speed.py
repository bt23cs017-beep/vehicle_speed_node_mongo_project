import cv2
import numpy as np

def build_homography(image_points, real_world_points):
    src = np.float32(image_points)
    dst = np.float32(real_world_points)
    return cv2.getPerspectiveTransform(src, dst)

def pixel_to_world(point, H):
    p = np.array([[[point[0], point[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(p, H)[0][0]
    return float(mapped[0]), float(mapped[1])

def speed_kmph(distance_m, delta_seconds):
    if delta_seconds <= 0:
        return 0.0
    return (distance_m / delta_seconds) * 3.6
