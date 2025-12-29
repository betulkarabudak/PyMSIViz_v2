# core/roi_manager.py
import numpy as np
class ROIManager:
    def __init__(self):
        self.saved_rois = {}          # name -> {"mask": ndarray, "z": int}
        self.last_roi_mask = None     # ndarray (full image)
        self.last_roi_z = None        # int

    # ---------------- Basic API ----------------
    def add_roi(self, *, name, mask, z):
        self.saved_rois[name] = {
            "mask": mask,
            "z": int(z)
        }

    def get_roi(self, name):
        return self.saved_rois.get(name)

    def remove_roi(self, name):
        if name in self.saved_rois:
            del self.saved_rois[name]

    def clear(self):
        self.saved_rois.clear()
        self.last_roi_mask = None
        self.last_roi_z = None

    # ---------------- Helpers ----------------
    def has_rois(self):
        return len(self.saved_rois) > 0

    def names(self):
        return list(self.saved_rois.keys())


