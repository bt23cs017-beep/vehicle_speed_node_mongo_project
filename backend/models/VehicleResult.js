const mongoose = require("mongoose");

const vehicleResultSchema = new mongoose.Schema({
  videoId: { type: String, required: true },
  vehicleId: { type: Number, required: true },
  type: { type: String, default: "vehicle" },
  speedKmph: { type: Number, default: null },
  crossedRoi: { type: Boolean, default: false },
  direction: { type: String, default: "unknown" },
  crossingFrame: { type: Number, default: null },
  timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model("VehicleResult", vehicleResultSchema);
