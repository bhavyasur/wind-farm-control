from floris import FlorisModel
import numpy as np

class GaussWakeModel:
    def __init__(self):
        self.fm = FlorisModel("configs/farm_setup.yml")
        self.fm.set_wake_model(
            velocity_model="gauss",
            deflection_model="gauss"
        )

    @property
    def param_ranges(self):
        return {
            "ka": (0.1, 0.3),
            "kb": (0.1, 0.5),
            "alpha": (0.1, 0.2)
        }

    def sample_params(self):
        return {k: np.random.uniform(*v) for k, v in self.param_ranges.items()}

    def apply_params(self, params):
        vmod = self.fm.wake.velocity_model
        dmod = self.fm.wake.deflection_model
        for k, v in params.items():
            if hasattr(vmod, k):
                setattr(vmod, k, v)
            if hasattr(dmod, k):
                setattr(dmod, k, v)
