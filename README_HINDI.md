# Vehicle Counting & Speed Estimation from CCTV

Exact project method:
YOLOv8 + DeepSORT + ROI Line Crossing + Homography + Speed Estimation

Architecture:
Mobile/CCTV Video -> Node.js/Express -> Python CV Processor -> Processed Video
                                  -> MongoDB (vehicle results)

## Folder structure

- backend/ : Node.js + Express + MongoDB
- python_cv/ : YOLO/DeepSORT/ROI/Homography/speed processing
- frontend/ : simple web UI
- uploads/input/ : input mobile/CCTV videos
- uploads/output/ : processed videos
- models/ : YOLO weights
- datasets/ : training dataset
- config/ : ROI and homography calibration
- logs/ : processing logs

## Run order

1. Install Node.js, Python and MongoDB.
2. Open this folder in VS Code.
3. `cd backend`
4. `npm install`
5. Copy `.env.example` to `.env`
6. Put a test video in `uploads/input/`.
6. Run `RUN_3_PYTHON_SETUP.bat` to install the Python CV dependencies.
7. Configure `config/homography_points.json` using the instructions in `README.md` before relying on speed values.
8. Start Node server with `RUN_2_START_SERVER.bat` or `cd backend` then `npm run dev`.
9. Open http://localhost:5000 and upload a traffic video.

IMPORTANT:
The Python processor is intentionally separated from Node.js. Node.js handles upload/API/database/UI, while Python handles computer vision.
