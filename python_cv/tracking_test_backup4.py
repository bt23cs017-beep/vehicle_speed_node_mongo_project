import cv2
import sys
import os
import csv
import math
import numpy as np
from collections import defaultdict, deque, Counter

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# INPUT / OUTPUT
# ============================================================
if len(sys.argv) < 2:
    print("ERROR: Input video path is required.")
    print("Usage: python tracking_test.py <input_video> [output_video]")
    sys.exit(1)

INPUT_VIDEO = sys.argv[1]
if len(sys.argv) >= 3:
    OUTPUT_VIDEO = sys.argv[2]
else:
    OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "uploads", "output", "processed_video.mp4")

# ============================================================
# YOLO MODEL (10-epoch model, NOT changed)
# ============================================================
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "runs", "detect", "runs", "train",
    "ua_detrac_yolov8n_30ep", "weights", "best_fixed.pt"
)

CLASS_NAMES = {0: "truck", 1: "car", 2: "van", 3: "bus"}

# ============================================================
# SETTINGS
# ============================================================
CONFIDENCE = 0.12

# Class ke hisaab se minimum confidence (false boxes kam karne ke liye)
CLASS_MIN_CONF = {0: 0.15, 1: 0.12, 2: 0.12, 3: 0.25}

# HORIZONTAL ROI line (frame height ka fraction)
#   $env:ROI_POSITION="0.60"
ROI_POSITION = float(os.environ.get("ROI_POSITION", "0.50"))

# VERTICAL ROI line (frame width ka fraction)
#   $env:ROI_X_POSITION="0.40"
ROI_X_POSITION = float(os.environ.get("ROI_X_POSITION", "0.50"))

# Dono lines ke around band (pixels)
ROI_BAND = int(os.environ.get("ROI_BAND", "20"))

MAX_COAST_FRAMES = 5

SPEED_FRAME_INTERVAL = 3
MAX_SPEED_KMH = 180.0
SPEED_HISTORY_LENGTH = 8

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DEBUG_CSV = os.path.join(LOG_DIR, "roi_debug.csv")

# ============================================================
# START
# ============================================================
print("==============================================")
print("VEHICLE COUNTING + SPEED ESTIMATION PIPELINE")
print("==============================================")
print("Input :", INPUT_VIDEO)
print("Output:", OUTPUT_VIDEO)
print("Model :", MODEL_PATH)

if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_VIDEO)), exist_ok=True)

print()
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Loading DeepSORT...")
tracker = DeepSort(max_age=45, n_init=2, nms_max_overlap=1.0)

print()
print("Opening input video...")
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not fps or fps <= 0 or math.isnan(fps):
    fps = 25.0

print(f"Video Size: {width} x {height}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")

# ============================================================
# ROI LINES (horizontal + vertical)
# ============================================================
ROI_Y = int(height * ROI_POSITION)
ROI_X = int(width * ROI_X_POSITION)
print("ROI Line Y (horizontal):", ROI_Y)
print("ROI Line X (vertical)  :", ROI_X)
print("ROI Band  :", ROI_BAND)

# ============================================================
# HOMOGRAPHY
# ============================================================
SOURCE_POINTS = np.float32([
    [594 * width / 1280, 490 * height / 720],
    [890 * width / 1280, 490 * height / 720],
    [521 * width / 1280, 660 * height / 720],
    [36 * width / 1280, 660 * height / 720]
])
DESTINATION_POINTS = np.float32([
    [0.0, 0.0], [3.5, 0.0], [3.5, 4.5], [0.0, 4.5]
])
homography_matrix, _ = cv2.findHomography(SOURCE_POINTS, DESTINATION_POINTS)
if homography_matrix is None:
    raise RuntimeError("Could not calculate homography matrix.")
print("Homography calibration: READY")

# ============================================================
# STATE
# ============================================================
counted_vehicle_ids = set()
previous_bottom_y = {}
previous_center_x = {}
total_crossed = 0

total_yolo_detections = 0
total_confirmed_tracks = 0
unique_confirmed_ids = set()
overlap_track_frames = 0

min_bottom_seen = None
max_bottom_seen = None

class_votes = defaultdict(Counter)
position_history = defaultdict(lambda: deque(maxlen=20))
speed_history = defaultdict(lambda: deque(maxlen=SPEED_HISTORY_LENGTH))
vehicle_speeds = {}


def pixel_to_world(px, py):
    point = np.float32([[px, py]]).reshape(-1, 1, 2)
    t = cv2.perspectiveTransform(point, homography_matrix)
    return float(t[0][0][0]), float(t[0][0][1])


def calculate_speed(track_id):
    history = position_history[track_id]
    if len(history) < 2:
        return vehicle_speeds.get(track_id)

    cur_frame, cur_x, cur_y = history[-1]
    old_item = None
    for item in reversed(history):
        if (cur_frame - item[0]) >= SPEED_FRAME_INTERVAL:
            old_item = item
            break
    if old_item is None:
        return vehicle_speeds.get(track_id)

    old_frame, old_x, old_y = old_item
    distance = math.sqrt((cur_x - old_x) ** 2 + (cur_y - old_y) ** 2)
    time_seconds = (cur_frame - old_frame) / fps
    if time_seconds <= 0:
        return vehicle_speeds.get(track_id)

    speed_kmh = min(distance / time_seconds * 3.6, MAX_SPEED_KMH)
    speed_history[track_id].append(speed_kmh)
    smoothed = sum(speed_history[track_id]) / len(speed_history[track_id])
    vehicle_speeds[track_id] = smoothed
    return smoothed


# ============================================================
# OUTPUT VIDEO + DEBUG CSV
# ============================================================
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not out.isOpened():
    raise RuntimeError("Could not create output video.")

csv_file = open(DEBUG_CSV, "w", newline="", encoding="utf-8")
csv_w = csv.writer(csv_file)
csv_w.writerow([
    "frame", "id", "type", "x1", "y1", "x2", "y2", "bottom_y", "center_x",
    "prev_bottom_y", "prev_center_x", "ROI_Y", "ROI_X", "ROI_BAND",
    "h_overlap", "v_overlap", "crossed_down", "crossed_up",
    "crossed_right", "crossed_left", "time_since_update",
    "counted_now", "already_counted"
])

# ============================================================
# PROCESS VIDEO
# ============================================================
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # ---------------- YOLO ----------------
    results = model.predict(frame, conf=CONFIDENCE, imgsz=960, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in CLASS_NAMES:
                continue
            confidence = float(box.conf[0])
            if confidence < CLASS_MIN_CONF.get(class_id, CONFIDENCE):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            detections.append(([x1, y1, bw, bh], confidence, CLASS_NAMES[class_id]))
            total_yolo_detections += 1

    # ---------------- DeepSORT ----------------
    tracks = tracker.update_tracks(detections, frame=frame)

    # ---------------- ROI bands (horizontal + vertical) ----------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, ROI_Y - ROI_BAND), (width, ROI_Y + ROI_BAND), (0, 255, 255), -1)
    cv2.rectangle(overlay, (ROI_X - ROI_BAND, 0), (ROI_X + ROI_BAND, height), (255, 255, 0), -1)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    cv2.line(frame, (0, ROI_Y), (width, ROI_Y), (0, 0, 255), 3)
    cv2.line(frame, (ROI_X, 0), (ROI_X, height), (255, 0, 0), 3)

    cv2.putText(frame, "HORIZONTAL ROI", (20, max(30, ROI_Y - ROI_BAND - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "VERTICAL ROI", (min(width - 190, ROI_X + ROI_BAND + 8), 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    current_vehicle_count = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        tsu = track.time_since_update
        if tsu > MAX_COAST_FRAMES:
            continue

        track_id = track.track_id
        total_confirmed_tracks += 1
        unique_confirmed_ids.add(track_id)

        l, t, r, b = track.to_ltrb()
        x1 = int(max(0, min(width - 1, l)))
        y1 = int(max(0, min(height - 1, t)))
        x2 = int(max(0, min(width - 1, r)))
        y2 = int(max(0, min(height - 1, b)))
        if x2 <= x1 or y2 <= y1:
            continue

        # ---- vehicle type (majority vote) ----
        if tsu == 0:
            dc = track.get_det_class()
            if dc is not None:
                class_votes[track_id][dc] += 1
        if track_id in class_votes and len(class_votes[track_id]) > 0:
            vehicle_type = class_votes[track_id].most_common(1)[0][0]
        else:
            vehicle_type = "vehicle"

        current_vehicle_count += 1

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        bottom_y = y2

        if min_bottom_seen is None or bottom_y < min_bottom_seen:
            min_bottom_seen = bottom_y
        if max_bottom_seen is None or bottom_y > max_bottom_seen:
            max_bottom_seen = bottom_y

        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 0), -1)
        cv2.circle(frame, (center_x, bottom_y), 6, (255, 0, 255), -1)

        # ---- speed (sirf real detection wale frames se) ----
        if tsu == 0:
            wx, wy = pixel_to_world(center_x, bottom_y)
            position_history[track_id].append((frame_number, wx, wy))
        speed_kmh = calculate_speed(track_id)

        # ====================================================
        # ROI COUNTING (HORIZONTAL + VERTICAL)
        # ====================================================
        previous_y = previous_bottom_y.get(track_id)
        previous_x = previous_center_x.get(track_id)

        # Horizontal line
        h_overlap = (y1 <= ROI_Y + ROI_BAND) and (y2 >= ROI_Y - ROI_BAND)
        crossed_down = (previous_y is not None and previous_y < ROI_Y and bottom_y >= ROI_Y)
        crossed_up = (previous_y is not None and previous_y > ROI_Y and bottom_y <= ROI_Y)

        # Vertical line
        v_overlap = (x1 <= ROI_X + ROI_BAND) and (x2 >= ROI_X - ROI_BAND)
        crossed_right = (previous_x is not None and previous_x < ROI_X and center_x >= ROI_X)
        crossed_left = (previous_x is not None and previous_x > ROI_X and center_x <= ROI_X)

        touching_any = h_overlap or v_overlap
        crossed_any = crossed_down or crossed_up or crossed_right or crossed_left

        if touching_any:
            overlap_track_frames += 1

        already_counted = track_id in counted_vehicle_ids
        counted_now = False

        if (not already_counted) and crossed_any:
            counted_vehicle_ids.add(track_id)
            total_crossed += 1
            counted_now = True

            if crossed_down:
                direction = "down (horizontal line)"
            elif crossed_up:
                direction = "up (horizontal line)"
            elif crossed_right:
                direction = "right (vertical line)"
            elif crossed_left:
                direction = "left (vertical line)"
            elif h_overlap:
                direction = "touch (horizontal line)"
            else:
                direction = "touch (vertical line)"

            sp_txt = f"{speed_kmh:.1f} km/h" if speed_kmh is not None else "--"
            print(
                "\n"
                "====================================\n"
                "ROI CROSSING DETECTED\n"
                f"Frame        : {frame_number}\n"
                f"Vehicle ID   : {track_id}\n"
                f"Vehicle Type : {vehicle_type}\n"
                f"Direction    : {direction}\n"
                f"Speed        : {sp_txt}\n"
                f"Box x1,x2    : {x1}, {x2}\n"
                f"Box y1,y2    : {y1}, {y2}\n"
                f"ROI Y / X    : {ROI_Y} / {ROI_X}\n"
                f"TOTAL COUNT  : {total_crossed}\n"
                "===================================="
            )

        previous_bottom_y[track_id] = bottom_y
        previous_center_x[track_id] = center_x

        csv_w.writerow([
            frame_number, track_id, vehicle_type, x1, y1, x2, y2, bottom_y, center_x,
            previous_y if previous_y is not None else "",
            previous_x if previous_x is not None else "",
            ROI_Y, ROI_X, ROI_BAND,
            int(h_overlap), int(v_overlap), int(crossed_down), int(crossed_up),
            int(crossed_right), int(crossed_left), tsu,
            int(counted_now), int(already_counted)
        ])

        # ---- draw box ----
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if track_id in counted_vehicle_ids:
            badge = "COUNTED"
            (bdw, bdh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            by = min(height - 8, y2 + bdh + 10)
            cv2.rectangle(frame, (x1, by - bdh - 6), (x1 + bdw + 8, by + 4), (0, 255, 255), -1)
            cv2.putText(frame, badge, (x1 + 4, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # ---- label ----
        sp_label = f"{speed_kmh:.1f} km/h" if speed_kmh is not None else "-- km/h"
        label = f"ID: {track_id} | {vehicle_type} | {sp_label}"

        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
        label_x = max(0, x1)
        label_y = max(y1 - 10, th + 10)

        cv2.rectangle(frame, (label_x, label_y - th - 8), (label_x + tw + 8, label_y + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

    # ---------------- info text ----------------
    cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles in Frame: {current_vehicle_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Crossed: {total_crossed}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.putText(frame, "Speed: Homography Based", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    out.write(frame)

    percentage = (frame_number / total_frames * 100) if total_frames > 0 else 0
    print(f"\rProcessing: {frame_number}/{total_frames} ({percentage:.1f}%)", end="")

# ============================================================
# RELEASE
# ============================================================
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

# ============================================================
# FINAL RESULT  (Node.js yeh lines parse karta hai - wording same)
# ============================================================
print()
print("==============================================")
print("VEHICLE PROCESSING COMPLETE")
print("==============================================")
print("Input:", INPUT_VIDEO)
print("Output:", OUTPUT_VIDEO)
print("Total Unique Vehicles Crossed:", total_crossed)
print("Total YOLO Vehicle Detections:", total_yolo_detections)
print("Total Confirmed DeepSORT Tracks:", total_confirmed_tracks)
print("Speed Estimation:", "Homography Based")
print("----------------------------------------------")
print("DIAGNOSTIC")
print("Unique track IDs seen   :", len(unique_confirmed_ids))
print("ROI_Y / ROI_X / ROI_BAND:", ROI_Y, "/", ROI_X, "/", ROI_BAND)
print("Vehicle bottom_y range  :", min_bottom_seen, "to", max_bottom_seen)
print("Track-frames on ROI band:", overlap_track_frames)
print("Debug CSV               :", DEBUG_CSV)
if total_crossed == 0:
    print("WARNING: Koi vehicle ROI lines tak nahi pahuncha.")
print("==============================================")



