const express = require("express");
const router = express.Router();
const VehicleResult = require("../models/VehicleResult");

router.get("/:videoId", async (req, res) => {
  try {
    const results = await VehicleResult.find({ videoId: req.params.videoId })
      .sort({ vehicleId: 1 });
    res.json({ success: true, results });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
