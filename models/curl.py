from floris import FlorisModel
import numpy as np

class CurlWakeModel:
    def __init__(self):
        self.fm = FlorisModel("configs/farm_setup.yml")
        self.fm.set_wake_model(
            velocity_model="curl",
            deflection_model="curl"
        )

    @property
    def param_ranges(self):
        return {
            "we": (0.6, 1.1),
            "wm": (0.1, 0.5),
            "me": (0.2, 0.6)
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
