from floris import FlorisModel
import numpy as np

class JensenWakeModel:
    def __init__(self):
        self.fm = FlorisModel("configs/farm_setup.yaml")
        self.fm.set_wake_model(
            velocity_model="jensen",
            deflection_model="jensen"
        )

    @property
    def param_ranges(self):
        return {
            "alpha": (0.05, 0.10),
            "beta": (0.05, 0.20)
        }

    def sample_params(self):
        return {k: np.random.uniform(*v) for k, v in self.param_ranges.items()}

    def apply_params(self, params):
        wake = self.fm.wake
        for k, v in params.items():
            if hasattr(wake.velocity_model, k):
                setattr(wake.velocity_model, k, v)
            if hasattr(wake.deflection_model, k):
                setattr(wake.deflection_model, k, v)
