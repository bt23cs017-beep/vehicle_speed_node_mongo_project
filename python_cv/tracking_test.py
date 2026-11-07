import cv2
import sys
import os
import csv
import json
import math
import numpy as np
from collections import defaultdict, deque, Counter

from ultralytics import YOLO
from deep_sort_compat import DeepSort

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "config", "homography_points.json")

with open(CALIBRATION_PATH, "r", encoding="utf-8") as calibration_file:
    CALIBRATION_CONFIG = json.load(calibration_file)

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
OUTPUT_STEM = os.path.splitext(OUTPUT_VIDEO)[0]
RESULT_JSON = OUTPUT_STEM + "_results.json"
PROGRESS_JSON = OUTPUT_STEM + "_progress.json"
LIVE_PREVIEW = OUTPUT_STEM + "_live.jpg"
LIVE_PREVIEW_TEMP = OUTPUT_STEM + "_live.tmp.jpg"

# ============================================================
# YOLO MODEL
# ============================================================
MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")
)
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATH)

# ============================================================
# SETTINGS
# ============================================================
CONFIDENCE = 0.25

# Filter by class name, not class ID: COCO and the custom checkpoint use different IDs.
CLASS_MIN_CONF = {
    "car": 0.25,
    "motorcycle": 0.25,
    "bus": 0.25,
    "truck": 0.25,
    "van": 0.25
}

# Fallback horizontal ROI position until a line is configured in calibration JSON.
ROI_POSITION = float(os.environ.get("ROI_POSITION", "0.50"))
ROI_BAND = int(os.environ.get("ROI_BAND", "20"))

MAX_COAST_FRAMES = 5

try:
    DETECTION_FRAME_STRIDE = max(1, int(os.environ.get("DETECTION_FRAME_STRIDE", "4")))
except ValueError:
    DETECTION_FRAME_STRIDE = 4

SPEED_FRAME_INTERVAL = 8
MAX_SPEED_KMH = 200.0
LIVE_UPDATE_INTERVAL = 4

# ---- Pixel-displacement speed estimation (fallback / primary for uncalibrated) ----
# How many pixels correspond to 1 metre in the video.
# This is approximate and depends on camera angle & zoom.
# For a typical highway CCTV at ~10-15m height: 8-15 px/m is common.
# For the 426x240 test video ("short vid vehicle.mp4"): ~6 px/m works well.
try:
    PIXELS_PER_METER = float(os.environ.get("PIXELS_PER_METER", "6.0"))
except (TypeError, ValueError):
    PIXELS_PER_METER = 6.0
# Minimum frames between position samples for displacement speed
SPEED_SAMPLE_WINDOW = max(2, int(os.environ.get("SPEED_SAMPLE_WINDOW", "6")))
# How many recent position samples to keep per vehicle
SPEED_HISTORY_LENGTH = max(4, int(os.environ.get("SPEED_HISTORY_LENGTH", "12")))

try:
    MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))
except ValueError:
    MAX_FRAMES = 0

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
input_stem = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
DEBUG_CSV = os.environ.get(
    "ROI_DEBUG_CSV",
    os.path.join(LOG_DIR, f"{input_stem}_roi_debug.csv")
)

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
print("Detector classes:", model.names)

print("Loading DeepSORT...")
tracker = DeepSort(
    max_age=45,
    n_init=3,
    nms_max_overlap=1.0
)

print()
print("Opening input video...")
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
overlay_scale = max(0.35, min(1.0, min(width / 1280, height / 720)))
vehicle_text_scale = max(0.32, min(0.62, 0.76 * overlay_scale))
status_text_scale = max(0.36, min(0.55, 0.68 * overlay_scale))
gate_text_scale = max(0.32, min(0.46, 0.58 * overlay_scale))
overlay_thickness = max(1, round(2 * overlay_scale))
marker_radius = max(2, round(5 * overlay_scale))
if not fps or fps <= 0 or math.isnan(fps):
    fps = 25.0
gate_thickness = max(2, round(3 * overlay_scale))

print(f"Video Size: {width} x {height}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
source_long_side = max(width, height)
INFERENCE_SIZE = (
    448
    if source_long_side <= 480
    else max(416, min(960, (source_long_side // 32) * 32))
)
print("Inference size:", INFERENCE_SIZE)
print("Detection frame stride:", DETECTION_FRAME_STRIDE)

# ============================================================
# ROI LINE
# ============================================================
ROI_Y = int(height * ROI_POSITION)
ROI_X = width // 2
reference_size = CALIBRATION_CONFIG.get("reference_size", {})
reference_width = reference_size.get("width", 0)
reference_height = reference_size.get("height", 0)
roi_config = CALIBRATION_CONFIG.get("roi_line", {})
roi_values = [roi_config.get(key, 0) for key in ("x1", "y1", "x2", "y2")]
roi_is_configured = all(isinstance(value, (int, float)) for value in roi_values) and any(roi_values)
if roi_is_configured:
    scale_x = width / reference_width if reference_width else 1.0
    scale_y = height / reference_height if reference_height else 1.0
    ROI_START = (int(roi_values[0] * scale_x), int(roi_values[1] * scale_y))
    ROI_END = (int(roi_values[2] * scale_x), int(roi_values[3] * scale_y))
    roi_is_configured = ROI_START != ROI_END
if not roi_is_configured:
    ROI_START = (0, ROI_Y)
    ROI_END = (width - 1, ROI_Y)
ROI_Y = (ROI_START[1] + ROI_END[1]) // 2
ROI_X = (ROI_START[0] + ROI_END[0]) // 2
print("ROI line:", ROI_START, "to", ROI_END)
print("ROI Band  :", ROI_BAND)

# ============================================================
# SPEED GATES
# ============================================================
speed_gate_config = CALIBRATION_CONFIG.get("speed_gate", {})
line_1_fraction = float(speed_gate_config.get("line_1_y_fraction", 0.40))
line_2_fraction = float(speed_gate_config.get("line_2_y_fraction", 0.60))
if not (0.0 <= line_1_fraction <= 1.0):
    line_1_fraction = 0.40
if not (0.0 <= line_2_fraction <= 1.0):
    line_2_fraction = 0.60
SPEED_LINE_1 = ((0, int(height * line_1_fraction)), (width - 1, int(height * line_1_fraction)))
SPEED_LINE_2 = ((0, int(height * line_2_fraction)), (width - 1, int(height * line_2_fraction)))
try:
    distance_setting = os.environ.get(
        "SPEED_GATE_DISTANCE_M",
        speed_gate_config.get("distance_m")
    )
    SPEED_GATE_DISTANCE_M = float(distance_setting or 0.0)
except (TypeError, ValueError):
    SPEED_GATE_DISTANCE_M = 0.0
SPEED_CALIBRATED = (
    math.isfinite(SPEED_GATE_DISTANCE_M)
    and SPEED_GATE_DISTANCE_M > 0
    and SPEED_LINE_1[0][1] != SPEED_LINE_2[0][1]
)
print("Speed gate line 1:", SPEED_LINE_1)
print("Speed gate line 2:", SPEED_LINE_2)
print("Speed gate distance:", SPEED_GATE_DISTANCE_M if SPEED_CALIBRATED else "NOT CONFIGURED")

# ============================================================
# STATE
# ============================================================
counted_vehicle_ids = set()
previous_bottom_y = {}
previous_center_x = {}
vehicle_results = []
total_crossed = 0

total_yolo_detections = 0
total_confirmed_tracks = 0
unique_confirmed_ids = set()
overlap_track_frames = 0

min_bottom_seen = None
max_bottom_seen = None

class_votes = defaultdict(Counter)
previous_speed_positions = {}
speed_gate_crossings = {}
vehicle_speeds = {}
speed_gate_event_counts = {1: 0, 2: 0}
speed_pair_attempts = 0
speed_measurement_count = 0

# ---- Displacement-based speed tracking ----
# Stores deque of (frame_number, (cx, bottom_y)) for each track
position_history = defaultdict(lambda: deque(maxlen=SPEED_HISTORY_LENGTH))
displacement_speeds = {}  # track_id -> latest displacement speed in km/h


def crossed_segment(previous_point, current_point, line_start, line_end):
    line_x = line_end[0] - line_start[0]
    line_y = line_end[1] - line_start[1]
    previous_side = line_x * (previous_point[1] - line_start[1]) - line_y * (previous_point[0] - line_start[0])
    current_side = line_x * (current_point[1] - line_start[1]) - line_y * (current_point[0] - line_start[0])
    if previous_side == 0 or previous_side * current_side > 0:
        return False, False, None

    fraction = previous_side / (previous_side - current_side)
    crossing_x = previous_point[0] + fraction * (current_point[0] - previous_point[0])
    crossing_y = previous_point[1] + fraction * (current_point[1] - previous_point[1])
    line_length_squared = line_x * line_x + line_y * line_y
    projection = ((crossing_x - line_start[0]) * line_x
                  + (crossing_y - line_start[1]) * line_y) / line_length_squared
    return 0.0 <= projection <= 1.0, current_side > previous_side, fraction


def calculate_gate_speed(track_id, frame_number, current_point, previous_detection):
    global speed_pair_attempts, speed_measurement_count
    if not SPEED_CALIBRATED or previous_detection is None:
        return vehicle_speeds.get(track_id)

    previous_frame, previous_point = previous_detection
    line_events = []
    for gate_id, line in ((1, SPEED_LINE_1), (2, SPEED_LINE_2)):
        did_cross, direction, fraction = crossed_segment(
            previous_point, current_point, line[0], line[1]
        )
        if did_cross:
            speed_gate_event_counts[gate_id] += 1
            crossing_frame = previous_frame + fraction * (frame_number - previous_frame)
            line_events.append((crossing_frame, gate_id, direction))

    for crossing_frame, gate_id, direction in sorted(line_events):
        previous_crossing = speed_gate_crossings.get(track_id)
        if (
            previous_crossing is not None
            and previous_crossing["gate"] != gate_id
            and previous_crossing["direction"] == direction
        ):
            speed_pair_attempts += 1
            elapsed_seconds = abs(crossing_frame - previous_crossing["frame"]) / fps
            if elapsed_seconds > 0:
                speed_kmh = SPEED_GATE_DISTANCE_M / elapsed_seconds * 3.6
                if math.isfinite(speed_kmh) and 0 < speed_kmh <= MAX_SPEED_KMH:
                    vehicle_speeds[track_id] = speed_kmh
                    speed_measurement_count += 1

        speed_gate_crossings[track_id] = {
            "frame": crossing_frame,
            "gate": gate_id,
            "direction": direction
        }

    return vehicle_speeds.get(track_id)


def calculate_displacement_speed(track_id, frame_number, current_point):
    """Estimate speed from pixel displacement between position samples.

    Uses Euclidean distance between the oldest and newest recorded
    positions for the track, converts from pixels to metres using
    PIXELS_PER_METER, then to km/h using the frame timestamps.
    """
    history = position_history[track_id]
    history.append((frame_number, current_point))

    if len(history) < 2:
        return displacement_speeds.get(track_id)

    # Use the oldest sample that is at least SPEED_SAMPLE_WINDOW frames ago
    old_frame, old_point = history[0]
    frame_gap = frame_number - old_frame
    if frame_gap < SPEED_SAMPLE_WINDOW:
        return displacement_speeds.get(track_id)

    pixel_dist = math.hypot(
        current_point[0] - old_point[0],
        current_point[1] - old_point[1]
    )
    elapsed_seconds = frame_gap / fps
    if elapsed_seconds <= 0 or PIXELS_PER_METER <= 0:
        return displacement_speeds.get(track_id)

    distance_m = pixel_dist / PIXELS_PER_METER
    speed_kmh = (distance_m / elapsed_seconds) * 3.6

    if math.isfinite(speed_kmh) and 0 < speed_kmh <= MAX_SPEED_KMH:
        displacement_speeds[track_id] = speed_kmh

    return displacement_speeds.get(track_id)

def write_json_atomic(path, payload):
    temporary_path = f"{path}.{os.getpid()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file)
    try:
        os.replace(temporary_path, path)
    except PermissionError:
        try:
            os.remove(temporary_path)
        except OSError:
            pass


def publish_progress(status, frame_number, vehicles_in_frame=0):
    write_json_atomic(PROGRESS_JSON, {
        "status": status,
        "currentFrame": frame_number,
        "totalFrames": total_frames,
        "percent": round(frame_number / total_frames * 100, 1) if total_frames else 0,
        "vehiclesInFrame": vehicles_in_frame,
        "totalCrossed": total_crossed,
        "yoloDetections": total_yolo_detections,
        "speedCalibrated": SPEED_CALIBRATED or (len(displacement_speeds) > 0),
        "speedGateDistanceM": SPEED_GATE_DISTANCE_M if SPEED_CALIBRATED else None,
        "vehicles": vehicle_results
    })


def publish_live_preview(frame):
    success, encoded_frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if success:
        temporary_path = f"{LIVE_PREVIEW_TEMP}.{os.getpid()}"
        with open(temporary_path, "wb") as preview_file:
            preview_file.write(encoded_frame.tobytes())
        try:
            os.replace(temporary_path, LIVE_PREVIEW)
        except PermissionError:
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def roi_side(point):
    return ((ROI_END[0] - ROI_START[0]) * (point[1] - ROI_START[1])
            - (ROI_END[1] - ROI_START[1]) * (point[0] - ROI_START[0]))


def distance_to_roi(point):
    line_x = ROI_END[0] - ROI_START[0]
    line_y = ROI_END[1] - ROI_START[1]
    line_length_squared = line_x * line_x + line_y * line_y
    if line_length_squared == 0:
        return float("inf")
    projection = ((point[0] - ROI_START[0]) * line_x
                  + (point[1] - ROI_START[1]) * line_y) / line_length_squared
    projection = min(1.0, max(0.0, projection))
    nearest_x = ROI_START[0] + projection * line_x
    nearest_y = ROI_START[1] + projection * line_y
    return math.hypot(point[0] - nearest_x, point[1] - nearest_y)


def crossed_roi(previous_point, current_point):
    did_cross, direction, _ = crossed_segment(
        previous_point, current_point, ROI_START, ROI_END
    )
    return did_cross, direction


# ============================================================
# OUTPUT VIDEO + DEBUG CSV
# ============================================================
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)
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
publish_progress("processing", 0)

# ============================================================
# PROCESS VIDEO
# ============================================================
frame_number = 0
processed_frames = 0
cached_tracks = []

while True:
    if MAX_FRAMES > 0 and frame_number >= MAX_FRAMES:
        break
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    processed_frames += 1
    detection_frame = frame_number == 1 or (frame_number - 1) % DETECTION_FRAME_STRIDE == 0

    # ---------------- YOLO ----------------
    if detection_frame:
        results = model.predict(frame, conf=CONFIDENCE, imgsz=INFERENCE_SIZE, verbose=False)
    else:
        results = []

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_names = model.names
            if isinstance(class_names, dict):
                class_name = class_names.get(class_id)
            else:
                class_name = class_names[class_id] if class_id < len(class_names) else None
            if class_name is None:
                continue
            class_name = str(class_name).strip().lower()
            if class_name not in CLASS_MIN_CONF:
                continue
            confidence = float(box.conf[0])
            if confidence < CLASS_MIN_CONF[class_name]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            detections.append(([x1, y1, bw, bh], confidence, class_name))
            total_yolo_detections += 1

    # ---------------- DeepSORT ----------------
    if detection_frame:
        cached_tracks = tracker.update_tracks(detections, frame=frame)
    tracks = cached_tracks

    # ---------------- ROI band ----------------
    roi_matches_gate_1 = ROI_START == SPEED_LINE_1[0] and ROI_END == SPEED_LINE_1[1]
    roi_matches_gate_2 = ROI_START == SPEED_LINE_2[0] and ROI_END == SPEED_LINE_2[1]
    if not roi_matches_gate_1 and not roi_matches_gate_2:
        cv2.line(frame, ROI_START, ROI_END, (0, 255, 255), overlay_thickness)
        cv2.putText(frame, "COUNT ROI", (20, max(30, ROI_Y - ROI_BAND - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, gate_text_scale, (0, 255, 255), 1)
    cv2.line(frame, SPEED_LINE_1[0], SPEED_LINE_1[1], (0, 0, 255), gate_thickness)
    cv2.line(frame, SPEED_LINE_2[0], SPEED_LINE_2[1], (0, 0, 255), gate_thickness)
    cv2.putText(frame, "GATE 1", (4, max(14, SPEED_LINE_1[0][1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, gate_text_scale, (0, 0, 255), 1)
    gate_2_label = "GATE 2 / COUNT" if roi_matches_gate_2 else "GATE 2"
    cv2.putText(frame, gate_2_label, (4, max(14, SPEED_LINE_2[0][1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, gate_text_scale, (0, 0, 255), 1)

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
        if detection_frame and tsu == 0:
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

        cv2.circle(frame, (center_x, center_y), marker_radius, (255, 255, 0), -1)
        cv2.circle(frame, (center_x, bottom_y), marker_radius, (255, 0, 255), -1)

        # ---- speed estimation ----
        current_point = (center_x, bottom_y)

        # Primary: two-gate crossing speed (most accurate when calibrated)
        gate_speed = vehicle_speeds.get(track_id)
        if detection_frame:
            previous_speed_detection = previous_speed_positions.get(track_id)
            gate_speed = calculate_gate_speed(
                track_id,
                frame_number,
                current_point,
                previous_speed_detection
            )
            previous_speed_positions[track_id] = (frame_number, current_point)

        # Fallback: pixel-displacement speed (works for every tracked vehicle)
        disp_speed = displacement_speeds.get(track_id)
        if detection_frame:
            disp_speed = calculate_displacement_speed(
                track_id, frame_number, current_point
            )

        # Gate speed wins if available, otherwise use displacement speed
        speed_kmh = gate_speed if gate_speed is not None else disp_speed

        if speed_kmh is not None:
            for result in vehicle_results:
                if result["vehicleId"] == int(track_id):
                    result["speedKmph"] = round(speed_kmh, 2)
                    result["speedMethod"] = "gate" if gate_speed is not None else "displacement"

        # ====================================================
        # ROI COUNTING (HORIZONTAL + VERTICAL)
        # ====================================================
        previous_y = previous_bottom_y.get(track_id)
        previous_x = previous_center_x.get(track_id)

        previous_point = (previous_x, previous_y) if previous_x is not None and previous_y is not None else None
        touching_any = distance_to_roi(current_point) <= ROI_BAND
        crossed_any, positive_direction = (
            crossed_roi(previous_point, current_point)
            if detection_frame and previous_point is not None
            else (False, False)
        )
        h_overlap = touching_any
        v_overlap = False
        crossed_down = crossed_any and positive_direction
        crossed_up = crossed_any and not positive_direction
        crossed_right = False
        crossed_left = False

        if touching_any:
            overlap_track_frames += 1

        already_counted = track_id in counted_vehicle_ids
        counted_now = False

        if (not already_counted) and crossed_any:
            counted_vehicle_ids.add(track_id)
            total_crossed += 1
            counted_now = True

            direction = "forward" if positive_direction else "reverse"
            speed_method = (
                "gate" if gate_speed is not None
                else ("displacement" if disp_speed is not None else None)
            )
            vehicle_results.append({
                "vehicleId": int(track_id),
                "type": vehicle_type,
                "speedKmph": round(speed_kmh, 2) if speed_kmh is not None else None,
                "speedMethod": speed_method,
                "crossedRoi": True,
                "direction": direction,
                "crossingFrame": frame_number
            })

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
                f"ROI line     : {ROI_START} to {ROI_END}\n"
                f"TOTAL COUNT  : {total_crossed}\n"
                "===================================="
            )

        if detection_frame:
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), overlay_thickness)

        if track_id in counted_vehicle_ids and width >= 640:
            badge = "COUNTED"
            (bdw, bdh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, vehicle_text_scale, 1)
            by = min(height - 2, y2 + bdh + 4)
            cv2.rectangle(frame, (x1, max(0, by - bdh - 3)), (min(width - 1, x1 + bdw + 4), by), (0, 255, 255), -1)
            cv2.putText(frame, badge, (x1 + 2, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, vehicle_text_scale, (0, 0, 0), 1)

        # ---- label ----
        label = (
            f"{vehicle_type} {speed_kmh:.0f} km/h"
            if speed_kmh is not None
            else vehicle_type
        )

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, vehicle_text_scale, 1)
        padding = max(2, round(4 * overlay_scale))
        label_x = min(max(0, x1), max(0, width - tw - padding * 2))
        label_top = max(0, y1 - th - baseline - padding * 2)
        label_y = min(height - 1, label_top + th + padding)
        cv2.rectangle(
            frame,
            (label_x, label_top),
            (min(width - 1, label_x + tw + padding * 2), min(height - 1, label_y + baseline + padding)),
            (0, 0, 0),
            -1
        )
        cv2.putText(frame, label, (label_x + padding, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, vehicle_text_scale, (0, 255, 0), 1, cv2.LINE_AA)

    # ---------------- info text ----------------
    distance_status = f"Gate: {SPEED_GATE_DISTANCE_M:.1f}m" if SPEED_CALIBRATED else f"Scale: {PIXELS_PER_METER:.1f}px/m"
    status_text = f"F {frame_number}/{total_frames} | Cars {current_vehicle_count} | Count {total_crossed} | {distance_status}"
    (status_width, status_height), status_baseline = cv2.getTextSize(
        status_text, cv2.FONT_HERSHEY_SIMPLEX, status_text_scale, 1
    )
    cv2.rectangle(frame, (0, 0), (min(width - 1, status_width + 8), status_height + status_baseline + 6), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (4, status_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX, status_text_scale, (255, 255, 255), 1, cv2.LINE_AA)

    out.write(frame)

    percentage = (frame_number / total_frames * 100) if total_frames > 0 else 0
    print(f"\rProcessing: {frame_number}/{total_frames} ({percentage:.1f}%)", end="")
    if processed_frames % LIVE_UPDATE_INTERVAL == 0 or frame_number == total_frames:
        publish_live_preview(frame)
        publish_progress("processing", frame_number, current_vehicle_count)

# ============================================================
# RELEASE
# ============================================================
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

for vehicle in vehicle_results:
    vid = vehicle["vehicleId"]
    gate_spd = vehicle_speeds.get(str(vid), vehicle_speeds.get(vid))
    disp_spd = displacement_speeds.get(str(vid), displacement_speeds.get(vid))
    # Gate speed takes priority
    final_speed = gate_spd if gate_spd is not None else disp_spd
    vehicle["speedKmph"] = round(final_speed, 2) if final_speed is not None else None
    vehicle["speedMethod"] = (
        "gate" if gate_spd is not None
        else ("displacement" if disp_spd is not None else None)
    )

summary = {
    "totalCrossed": total_crossed,
    "yoloDetections": total_yolo_detections,
    "uniqueTracks": len(unique_confirmed_ids),
    "speedCalibrated": SPEED_CALIBRATED or (len(displacement_speeds) > 0),
    "speedMethod": "two_gate_crossing + displacement_fallback",
    "pixelsPerMeter": PIXELS_PER_METER,
    "speedGateDistanceM": SPEED_GATE_DISTANCE_M if SPEED_CALIBRATED else None,
    "speedGateCrossings": {str(gate): count for gate, count in speed_gate_event_counts.items()},
    "speedPairAttempts": speed_pair_attempts,
    "speedMeasurements": speed_measurement_count,
    "displacementSpeedCount": len(displacement_speeds),
    "vehicleSpeedsKmph": {
        str(track_id): round(speed, 2)
        for track_id, speed in {**displacement_speeds, **vehicle_speeds}.items()
    },
    "speedLines": {
        "line1": {"x1": SPEED_LINE_1[0][0], "y1": SPEED_LINE_1[0][1], "x2": SPEED_LINE_1[1][0], "y2": SPEED_LINE_1[1][1]},
        "line2": {"x1": SPEED_LINE_2[0][0], "y1": SPEED_LINE_2[0][1], "x2": SPEED_LINE_2[1][0], "y2": SPEED_LINE_2[1][1]}
    },
    "roiLine": {"x1": ROI_START[0], "y1": ROI_START[1], "x2": ROI_END[0], "y2": ROI_END[1]},
    "vehicles": vehicle_results
}
with open(RESULT_JSON, "w", encoding="utf-8") as results_file:
    json.dump(summary, results_file, indent=2)
publish_progress("finalizing", frame_number)

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
print("Speed Estimation:", "Two-line gate" if SPEED_CALIBRATED else "Not calibrated")
print("----------------------------------------------")
print("DIAGNOSTIC")
print("Unique track IDs seen   :", len(unique_confirmed_ids))
print("ROI line / band        :", ROI_START, "to", ROI_END, "/", ROI_BAND)
print("Vehicle bottom_y range  :", min_bottom_seen, "to", max_bottom_seen)
print("Track-frames on ROI band:", overlap_track_frames)
print("Debug CSV               :", DEBUG_CSV)
print("Results JSON            :", RESULT_JSON)
print("Progress JSON           :", PROGRESS_JSON)
if total_crossed == 0:
    print("WARNING: Koi vehicle ROI lines tak nahi pahuncha.")
print("==============================================")






