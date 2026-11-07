const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");
const path = require("path");
const fs = require("fs");
const multer = require("multer");
const { spawn, execFile } = require("child_process");
const dns = require('dns');


dns.setServers(['8.8.8.8', '8.8.4.4']);

// Your existing code continues...

// ============================================================
// APP SETUP
// ============================================================

const app = express();

dotenv.config({
    path: path.join(__dirname, ".env")
});

const PORT = process.env.PORT || 5000;

// ============================================================
// DIRECTORIES
// ============================================================

const rootDir = path.join(
    __dirname,
    ".."
);

const frontendDir = path.join(
    rootDir,
    "frontend"
);

const uploadsDir = path.join(
    rootDir,
    "uploads"
);

const inputDir = path.join(
    uploadsDir,
    "input"
);

const outputDir = path.join(
    uploadsDir,
    "output"
);

const processingJobs = new Map();

const pythonDir = path.join(
    rootDir,
    "python_cv"
);

const pythonExecutable = process.env.PYTHON_EXECUTABLE
    ? path.resolve(rootDir, process.env.PYTHON_EXECUTABLE)
    : path.join(pythonDir, "venv", "Scripts", "python.exe");

const pythonScript = path.join(
    pythonDir,
    "tracking_test.py"
);

// ============================================================
// CREATE DIRECTORIES
// ============================================================

fs.mkdirSync(
    inputDir,
    {
        recursive: true
    }
);

fs.mkdirSync(
    outputDir,
    {
        recursive: true
    }
);

// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors());

app.use(
    express.json()
);

app.use("/api/results", require("./routes/results"));

// ============================================================
// SERVE FRONTEND
// ============================================================

app.use(
    express.static(
        frontendDir
    )
);

// ============================================================
// SERVE UPLOADED / PROCESSED VIDEOS
// ============================================================

app.use(
    "/uploads",
    express.static(
        uploadsDir,
        {
            acceptRanges: true,
            fallthrough: true,
            maxAge: "1h"
        }
    )
);

// ============================================================
// MONGODBconst dns = require('dns');

// ============================================================

console.log("MONGO_URI loaded:", process.env.MONGO_URI ? "YES" : "NO");
const mongoConnection = process.env.MONGO_URI
    ? mongoose.connect(process.env.MONGO_URI)
        .then(() => {
            console.log("MongoDB connected successfully");
            return true;
        })
        .catch((error) => {
            console.error("MongoDB connection error:", error.message);
            return false;
        })
    : Promise.resolve(false);

// ============================================================
// MULTER
// ============================================================

const storage =
    multer.diskStorage({

        destination:
            function (
                req,
                file,
                cb
            ) {

                cb(
                    null,
                    inputDir
                );

            },

        filename:
            function (
                req,
                file,
                cb
            ) {

                const extension =
                    path.extname(
                        file.originalname
                    );

                const baseName =
                    path.basename(
                        file.originalname,
                        extension
                    );

                const safeName =
                    baseName.replace(
                        /[^a-zA-Z0-9_-]/g,
                        "_"
                    );

                const finalName =
                    Date.now() +
                    "_" +
                    safeName +
                    extension;

                cb(
                    null,
                    finalName
                );

            }

    });

const upload =
    multer({

        storage: storage,

        limits: {

            fileSize:
                500 * 1024 * 1024

        },

        fileFilter:
            function (
                req,
                file,
                cb
            ) {

                const allowedExtensions = [

                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".webm"

                ];

                const extension =
                    path.extname(
                        file.originalname
                    ).toLowerCase();

                if (
                    allowedExtensions.includes(
                        extension
                    )
                ) {

                    cb(
                        null,
                        true
                    );

                }
                else {

                    cb(
                        new Error(
                            "Only video files are allowed."
                        )
                    );

                }

            }

    });

// ============================================================
// HEALTH CHECK
// ============================================================

app.get(
    "/api/health",
    (req, res) => {

        res.json({

            success: true,

            message:
                "Vehicle Speed Backend is running"

        });

    }
);

app.get("/api/video/progress/:filename", (req, res) => {
    const videoId = path.basename(req.params.filename);
    const job = processingJobs.get(videoId);
    if (!job) {
        return res.status(404).json({ success: false, message: "No processing job found." });
    }

    const baseName = path.basename(videoId, path.extname(videoId));
    const progressPath = path.join(outputDir, `${baseName}_raw_progress.json`);
    const previewFilename = `${baseName}_raw_live.jpg`;
    const previewPath = path.join(outputDir, previewFilename);
    let progress = {};

    if (fs.existsSync(progressPath)) {
        try {
            progress = JSON.parse(fs.readFileSync(progressPath, "utf8"));
        } catch (error) {
            console.warn("Could not read live processing progress:", error.message);
        }
    }

    const previewUrl = fs.existsSync(previewPath)
        ? `/uploads/output/${previewFilename}?v=${fs.statSync(previewPath).mtimeMs}`
        : null;

    res.set("Cache-Control", "no-store");
    return res.json({
        success: true,
        ...progress,
        status: job.status,
        error: job.error,
        previewUrl
    });
});

// ============================================================
// PROJECT METHOD
// ============================================================

app.get(
    "/api/project-method",
    (req, res) => {

        res.json({

            project:
                "Vehicle Counting and Speed Estimation from CCTV",

            method:
                "YOLOv8 + DeepSORT + ROI Counting + Two-Line Speed Gates",

            tools:
                "OpenCV + YOLOv8"

        });

    }
);

// ============================================================
// VIDEO UPLOAD API
// ============================================================

app.post(
    "/api/video/upload",
    upload.single("video"),
    (req, res) => {

        try {

            if (!req.file) {

                return res.status(400).json({

                    success: false,

                    message:
                        "No video file received."

                });

            }

            console.log(
                "Video uploaded:",
                req.file.filename
            );

            return res.json({

                success: true,

                message:
                    "Video uploaded successfully.",

                filename:
                    req.file.filename,

                originalName:
                    req.file.originalname,

                path:
                    "/uploads/input/" +
                    req.file.filename

            });

        }

        catch (error) {

            console.error(
                "Upload error:",
                error
            );

            return res.status(500).json({

                success: false,

                message:
                    "Video upload failed.",

                error:
                    error.message

            });

        }

    }
);

// ============================================================
// PROCESS VIDEO API
//
// Upload
//      ↓
// Python
// YOLOv8 + DeepSORT + ROI + Homography
//      ↓
// Raw OpenCV MP4
//      ↓
// FFmpeg H.264
//      ↓
// Browser playable MP4
//
// ============================================================

app.post(
    "/api/video/process",
    async (req, res) => {

        try {

            // ==================================================
            // GET FILENAME
            // ==================================================

            const filename =
                req.body.filename;

            if (!filename) {

                return res.status(400).json({

                    success: false,

                    message:
                        "Video filename is required."

                });

            }

            // ==================================================
            // SECURITY
            // ==================================================

            const safeFilename =
                path.basename(
                    filename
                );

            const existingJob = processingJobs.get(safeFilename);
            if (existingJob && existingJob.status === "processing") {
                return res.status(409).json({
                    success: false,
                    message: "This video is already being processed."
                });
            }

            const speedGateDistanceM = req.body.speedGateDistanceM === null
                || req.body.speedGateDistanceM === undefined
                || req.body.speedGateDistanceM === ""
                ? null
                : Number(req.body.speedGateDistanceM);

            if (
                speedGateDistanceM !== null
                && (!Number.isFinite(speedGateDistanceM)
                    || speedGateDistanceM <= 0
                    || speedGateDistanceM > 1000)
            ) {
                return res.status(400).json({
                    success: false,
                    message: "Measured gate distance must be between 0 and 1000 metres."
                });
            }

            // ==================================================
            // INPUT VIDEO
            // ==================================================

            const inputVideo =
                path.join(
                    inputDir,
                    safeFilename
                );

            if (
                !fs.existsSync(
                    inputVideo
                )
            ) {

                return res.status(404).json({

                    success: false,

                    message:
                        "Input video not found: " +
                        safeFilename

                });

            }

            // ==================================================
            // CHECK PYTHON
            // ==================================================

            if (
                !fs.existsSync(
                    pythonExecutable
                )
            ) {

                return res.status(500).json({

                    success: false,

                    message:
                        "Python virtual environment not found.",

                    path:
                        pythonExecutable

                });

            }

            // ==================================================
            // CHECK PYTHON SCRIPT
            // ==================================================

            if (
                !fs.existsSync(
                    pythonScript
                )
            ) {

                return res.status(500).json({

                    success: false,

                    message:
                        "tracking_test.py not found.",

                    path:
                        pythonScript

                });

            }

            // ==================================================
            // FILE NAMES
            // ==================================================

            const baseName =
                path.basename(
                    safeFilename,
                    path.extname(
                        safeFilename
                    )
                );

            // ==================================================
            // RAW OUTPUT
            // ==================================================

            const rawOutputFilename =
                baseName +
                "_raw.mp4";

            // ==================================================
            // FINAL OUTPUT
            // ==================================================

            const finalOutputFilename =
                baseName +
                "_processed.mp4";

            const resultsJsonFilename =
                baseName +
                "_raw_results.json";

            const progressJsonFilename =
                baseName +
                "_raw_progress.json";

            const livePreviewFilename =
                baseName +
                "_raw_live.jpg";

            const rawOutputVideo =
                path.join(
                    outputDir,
                    rawOutputFilename
                );

            const finalOutputVideo =
                path.join(
                    outputDir,
                    finalOutputFilename
                );

            const resultsJsonPath =
                path.join(
                    outputDir,
                    resultsJsonFilename
                );

            const progressJsonPath =
                path.join(
                    outputDir,
                    progressJsonFilename
                );

            const livePreviewPath =
                path.join(
                    outputDir,
                    livePreviewFilename
                );

            // ==================================================
            // REMOVE OLD OUTPUT FILES
            // ==================================================

            if (
                fs.existsSync(
                    rawOutputVideo
                )
            ) {

                fs.unlinkSync(
                    rawOutputVideo
                );

            }

            if (
                fs.existsSync(
                    finalOutputVideo
                )
            ) {

                fs.unlinkSync(
                    finalOutputVideo
                );

            }

            if (
                fs.existsSync(
                    resultsJsonPath
                )
            ) {

                fs.unlinkSync(
                    resultsJsonPath
                );

            }

            for (const stalePath of [progressJsonPath, livePreviewPath]) {
                if (fs.existsSync(stalePath)) {
                    fs.unlinkSync(stalePath);
                }
            }

            // ==================================================
            // PROCESSING LOG
            // ==================================================

            console.log("");

            console.log(
                "================================================"
            );

            console.log(
                "STARTING VEHICLE VIDEO PROCESSING"
            );

            console.log(
                "================================================"
            );

            console.log(
                "Input:",
                inputVideo
            );

            console.log(
                "Raw output:",
                rawOutputVideo
            );

            console.log(
                "Final output:",
                finalOutputVideo
            );

            console.log(
                "Python:",
                pythonExecutable
            );

            console.log(
                "================================================"
            );

            // ==================================================
            // RUN PYTHON
            // ==================================================

            const processingJob = {
                status: "processing",
                error: null
            };
            processingJobs.set(safeFilename, processingJob);

            const pythonEnvironment = { ...process.env };
            if (speedGateDistanceM !== null) {
                pythonEnvironment.SPEED_GATE_DISTANCE_M = String(speedGateDistanceM);
            } else {
                delete pythonEnvironment.SPEED_GATE_DISTANCE_M;
            }

            const pythonProcess =
                spawn(

                    pythonExecutable,

                    [
                        pythonScript,
                        inputVideo,
                        rawOutputVideo
                    ],

                    {
                        cwd: rootDir,
                        windowsHide: true,
                        env: pythonEnvironment
                    }

                );

            // ==================================================
            // PYTHON OUTPUT
            // ==================================================

            let pythonOutput = "";

            let pythonError = "";

            // ==================================================
            // PYTHON STDOUT
            // ==================================================

            pythonProcess.stdout.on(
                "data",
                (data) => {

                    const text =
                        data.toString();

                    pythonOutput += text;

                    console.log(
                        "[PYTHON]",
                        text.trim()
                    );

                }
            );

            // ==================================================
            // PYTHON STDERR
            // ==================================================

            pythonProcess.stderr.on(
                "data",
                (data) => {

                    const text =
                        data.toString();

                    pythonError += text;

                    console.log(
                        "[PYTHON STDERR]",
                        text.trim()
                    );

                }
            );

            // ==================================================
            // PYTHON START ERROR
            // ==================================================

            pythonProcess.on(
                "error",
                (error) => {

                    processingJob.status = "failed";
                    processingJob.error = error.message;

                    console.error(
                        "Python process error:",
                        error
                    );

                    if (
                        !res.headersSent
                    ) {

                        return res.status(500).json({

                            success: false,

                            message:
                                "Python process could not start.",

                            error:
                                error.message

                        });

                    }

                }
            );

            // ==================================================
            // PYTHON COMPLETE
            // ==================================================

            pythonProcess.on(
                "close",
                (code) => {

                    console.log("");

                    console.log(
                        "Python process finished."
                    );

                    console.log(
                        "Python exit code:",
                        code
                    );

                    // ==========================================
                    // PYTHON FAILED
                    // ==========================================

                    if (
                        code !== 0
                    ) {

                        processingJob.status = "failed";
                        processingJob.error = pythonError || pythonOutput || `Python exited with code ${code}.`;

                        if (
                            !res.headersSent
                        ) {

                            return res.status(500).json({

                                success: false,

                                message:
                                    "Python video processing failed.",

                                exitCode:
                                    code,

                                error:
                                    pythonError ||
                                    pythonOutput

                            });

                        }

                        return;

                    }

                    // ==========================================
                    // RAW OUTPUT CHECK
                    // ==========================================

                    if (
                        !fs.existsSync(
                            rawOutputVideo
                        )
                    ) {

                        processingJob.status = "failed";
                        processingJob.error = "Python completed without creating an output video.";

                        if (
                            !res.headersSent
                        ) {

                            return res.status(500).json({

                                success: false,

                                message:
                                    "Python completed but raw output video was not created.",

                                pythonOutput:
                                    pythonOutput

                            });

                        }

                        return;

                    }

                    // ==========================================
                    // RAW FILE SIZE
                    // ==========================================

                    const rawStats =
                        fs.statSync(
                            rawOutputVideo
                        );

                    console.log(
                        "Raw video size:",
                        rawStats.size,
                        "bytes"
                    );

                    if (
                        rawStats.size === 0
                    ) {

                        processingJob.status = "failed";
                        processingJob.error = "Raw processed video is empty.";

                        if (
                            !res.headersSent
                        ) {

                            return res.status(500).json({

                                success: false,

                                message:
                                    "Raw processed video is empty."

                            });

                        }

                        return;

                    }

                    // ==================================================
                    // LOAD STRUCTURED PROCESSING RESULTS
                    // ==================================================

                    let processingSummary;
                    try {
                        processingSummary = JSON.parse(
                            fs.readFileSync(resultsJsonPath, "utf8")
                        );
                        if (!Array.isArray(processingSummary.vehicles)) {
                            throw new Error("Results JSON has no vehicle list.");
                        }
                    } catch (resultsError) {
                        processingJob.status = "failed";
                        processingJob.error = resultsError.message;
                        return res.status(500).json({
                            success: false,
                            message: "Python finished without valid structured results.",
                            error: resultsError.message
                        });
                    }

                    const totalCrossed = Number.isInteger(processingSummary.totalCrossed)
                        ? processingSummary.totalCrossed
                        : processingSummary.vehicles.length;

                    // ==================================================
                    // COUNT LOG
                    // ==================================================

                    console.log("");

                    console.log(
                        "================================================"
                    );

                    console.log(
                        "VEHICLE COUNT RESULT"
                    );

                    console.log(
                        "Vehicles crossed:",
                        totalCrossed
                    );

                    console.log(
                        "================================================"
                    );

                    // ==================================================
                    // FFMPEG H.264 CONVERSION
                    // ==================================================

                    console.log("");

                    console.log(
                        "================================================"
                    );

                    console.log(
                        "STARTING FFMPEG H.264 CONVERSION"
                    );

                    console.log(
                        "================================================"
                    );

                    console.log(
                        "FFmpeg input:",
                        rawOutputVideo
                    );

                    console.log(
                        "FFmpeg output:",
                        finalOutputVideo
                    );

                    // ==================================================
                    // FFMPEG ARGUMENTS
                    // ==================================================

                    const ffmpegArgs = [

                        "-y",

                        "-i",
                        rawOutputVideo,

                        // --------------------------
                        // VIDEO
                        // --------------------------

                        "-c:v",
                        "libx264",

                        "-pix_fmt",
                        "yuv420p",

                        "-preset",
                        "veryfast",

                        "-crf",
                        "23",

                        "-movflags",
                        "+faststart",

                        // --------------------------
                        // AUDIO
                        // --------------------------

                        "-c:a",
                        "aac",

                        "-b:a",
                        "128k",

                        // --------------------------
                        // OUTPUT
                        // --------------------------

                        finalOutputVideo

                    ];

                    // ==================================================
                    // RUN FFMPEG
                    // ==================================================

                    execFile(
                        "ffmpeg",
                        ffmpegArgs,
                        {
                            windowsHide: true,

                            maxBuffer:
                                20 * 1024 * 1024

                        },
                        async (
                            ffmpegError,
                            stdout,
                            stderr
                        ) => {

                            // ==========================================
                            // FFMPEG ERROR
                            // ==========================================

                            if (
                                ffmpegError
                            ) {

                                processingJob.status = "failed";
                                processingJob.error = ffmpegError.message;

                                console.error(
                                    "FFmpeg conversion failed:"
                                );

                                console.error(
                                    ffmpegError.message
                                );

                                console.error(
                                    stderr
                                );

                                if (
                                    !res.headersSent
                                ) {

                                    return res.status(500).json({

                                        success: false,

                                        message:
                                            "FFmpeg H.264 conversion failed.",

                                        error:
                                            ffmpegError.message,

                                        ffmpegOutput:
                                            stderr

                                    });

                                }

                                return;

                            }

                            // ==========================================
                            // FFMPEG SUCCESS
                            // ==========================================

                            console.log(
                                "FFmpeg conversion completed."
                            );

                            // ==========================================
                            // FINAL OUTPUT CHECK
                            // ==========================================

                            if (
                                !fs.existsSync(
                                    finalOutputVideo
                                )
                            ) {

                                processingJob.status = "failed";
                                processingJob.error = "Final H.264 video was not created.";

                                if (
                                    !res.headersSent
                                ) {

                                    return res.status(500).json({

                                        success: false,

                                        message:
                                            "FFmpeg finished but final H.264 video was not created."

                                    });

                                }

                                return;

                            }

                            // ==========================================
                            // FINAL FILE SIZE
                            // ==========================================

                            const finalStats =
                                fs.statSync(
                                    finalOutputVideo
                                );

                            console.log(
                                "Final H.264 video size:",
                                finalStats.size,
                                "bytes"
                            );

                            if (
                                finalStats.size === 0
                            ) {

                                processingJob.status = "failed";
                                processingJob.error = "Final H.264 video is empty.";

                                if (
                                    !res.headersSent
                                ) {

                                    return res.status(500).json({

                                        success: false,

                                        message:
                                            "Final H.264 video is empty."

                                    });

                                }

                                return;

                            }

                            let databaseSaved = false;
                            let databaseError = null;
                            try {
                                const connected = await mongoConnection;
                                if (!connected || mongoose.connection.readyState !== 1) {
                                    throw new Error("MongoDB is not connected.");
                                }

                                const VehicleResultModel = require("./models/VehicleResult");
                                const vehicleDocuments = processingSummary.vehicles
                                    .map((vehicle) => ({
                                        videoId: safeFilename,
                                        vehicleId: Number(vehicle.vehicleId),
                                        type: vehicle.type || "vehicle",
                                        speedKmph: Number.isFinite(vehicle.speedKmph)
                                            ? vehicle.speedKmph
                                            : null,
                                        crossedRoi: Boolean(vehicle.crossedRoi),
                                        direction: vehicle.direction || "unknown",
                                        crossingFrame: Number.isInteger(vehicle.crossingFrame)
                                            ? vehicle.crossingFrame
                                            : null
                                    }))
                                    .filter((vehicle) => Number.isInteger(vehicle.vehicleId));

                                await VehicleResultModel.deleteMany({ videoId: safeFilename });
                                if (vehicleDocuments.length > 0) {
                                    await VehicleResultModel.insertMany(vehicleDocuments);
                                }
                                databaseSaved = true;
                            } catch (saveError) {
                                databaseError = saveError.message;
                                console.error("MongoDB result save failed:", databaseError);
                            }

                            // ==========================================
                            // DELETE RAW FILE
                            // ==========================================

                            try {

                                if (
                                    fs.existsSync(
                                        rawOutputVideo
                                    )
                                ) {

                                    fs.unlinkSync(
                                        rawOutputVideo
                                    );

                                }

                            }

                            catch (
                                deleteError
                            ) {

                                console.log(
                                    "Could not delete raw file:",
                                    deleteError.message
                                );

                            }

                            // ==========================================
                            // BROWSER VIDEO URL
                            // ==========================================

                            const browserVideoUrl =
                                "/uploads/output/" +
                                finalOutputFilename;

                            // ==========================================
                            // SUCCESS LOG
                            // ==========================================

                            console.log("");

                            console.log(
                                "================================================"
                            );

                            console.log(
                                "VIDEO PROCESSING COMPLETED SUCCESSFULLY"
                            );

                            console.log(
                                "================================================"
                            );

                            console.log(
                                "Final video:",
                                browserVideoUrl
                            );

                            console.log(
                                "Vehicles crossed:",
                                totalCrossed
                            );

                            console.log(
                                "================================================"
                            );

                            // ==========================================
                            // FINAL JSON RESPONSE
                            // ==========================================

                            processingJob.status = "complete";

                            if (
                                !res.headersSent
                            ) {

                                return res.json({

                                    success: true,

                                    message:
                                        "Video processed and converted to browser-compatible H.264 successfully.",

                                    // ------------------------------
                                    // INPUT
                                    // ------------------------------

                                    input:
                                        "/uploads/input/" +
                                        safeFilename,

                                    // ------------------------------
                                    // OUTPUT VIDEO
                                    // ------------------------------

                                    output:
                                        browserVideoUrl,

                                    outputH264:
                                        browserVideoUrl,

                                    videoUrl:
                                        browserVideoUrl,

                                    // ------------------------------
                                    // VEHICLE COUNT
                                    // ------------------------------

                                    totalCrossed:
                                        totalCrossed,

                                    vehiclesCrossed:
                                        totalCrossed,

                                    vehicleCount:
                                        totalCrossed,

                                    count:
                                        totalCrossed,

                                    videoId:
                                        safeFilename,

                                    vehicleResults:
                                        processingSummary.vehicles,

                                    speedCalibrated:
                                        Boolean(processingSummary.speedCalibrated),

                                    speedGateDistanceM:
                                        processingSummary.speedGateDistanceM,

                                    databaseSaved:
                                        databaseSaved,

                                    databaseError:
                                        databaseError

                                });

                            }

                        }
                    );

                }
            );

        }

        catch (error) {

            console.error(
                "Processing API error:",
                error
            );

            if (
                !res.headersSent
            ) {

                return res.status(500).json({

                    success: false,

                    message:
                        "Video processing failed.",

                    error:
                        error.message

                });

            }

        }

    }
);

// ============================================================
// ERROR HANDLER
// ============================================================

app.use(
    (
        error,
        req,
        res,
        next
    ) => {

        console.error(
            "Server error:",
            error.message
        );

        return res.status(500).json({

            success: false,

            message:
                error.message

        });

    }
);

// ============================================================
// FRONTEND FALLBACK
// ============================================================

app.get(
    "*",
    (req, res) => {

        res.sendFile(
            path.join(
                frontendDir,
                "index.html"
            )
        );

    }
);
// ============================================================
// START SERVER
// ============================================================

app.listen(
    PORT,
    () => {

        console.log(
            `Server running: http://localhost:${PORT}`
        );

    }
);