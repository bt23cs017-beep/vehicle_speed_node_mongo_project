# DeepSORT tracker module
# This file will keep the same ID for the same vehicle across video frames.

class VehicleTracker:
    def __init__(self):
        self.next_id = 1

    def update(self, detections):
        # Final implementation will connect a DeepSORT tracker here.
        return detections
