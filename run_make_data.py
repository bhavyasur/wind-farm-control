import yaml
import numpy as np
import pandas as pd

from models.jensen import JensenWakeModel
from models.gauss import GaussWakeModel
from models.curl import CurlWakeModel
from models.gch import GCHWakeModel

# --- CONFIG SELECTION ---

MODEL_NAME   = "gch"              # "jensen", "gauss", "curl", or "gch"
N_SAMPLES    = 1000               # number of datapoints to generate
ENV_FILE     = "configs/farm_1.yml" # path to your YAML

model_map = {
    "jensen": JensenWakeModel,
    "gauss":  GaussWakeModel,
    "curl":   CurlWakeModel,
    "gch":    GCHWakeModel,   # Gauss–Curl Hybrid
}

# --- YAML & layout helpers ---

def load_environment(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def sample_env_value(entry: dict) -> float:
    """
    samples scalar from a YAML entry that looks like:
      {fixed: value} or {min: lo, max: hi}
    """
    if "fixed" in entry:
        return float(entry["fixed"])
    return float(np.random.uniform(entry["min"], entry["max"]))

def generate_layout(num_turbines: int, spacing: float = 500.0):
    """
    Create a simple 1D row of turbines spaced along x,
    all at y = 0. Modify later if you want more complex layouts.
    """
    layout_x = [i * spacing for i in range(num_turbines)]
    layout_y = [0.0] * num_turbines
    return layout_x, layout_y

# --- MAIN PARAMETER SWEEP ---

def run_sweep(model_name: str, N: int, env: dict):
    # Instantiate appropriate model wrapper
    ModelClass = model_map[model_name]
    model = ModelClass()
    fm = model.fm

    # --------- Number of turbines & layout ----------
    if "num_turbines" in env:
        num_t = int(sample_env_value(env["num_turbines"]))
    else:
        # fallback: current layout length
        num_t = len(fm.layout_x)

    layout_x, layout_y = generate_layout(num_t)
    fm.set(layout_x=layout_x, layout_y=layout_y)

    rows = []

    for _ in range(N):
        # 1) Sample model tuning parameters
        params = model.sample_params()
        model.apply_params(params)

        # 2) Sample environment variables from YAML
        ws = sample_env_value(env["wind_speed"])
        wd = sample_env_value(env["wind_direction"])
        yaw = sample_env_value(env["yaw"])

        if "turbulence_intensity" in env:
            ti = sample_env_value(env["turbulence_intensity"])
            fm.set(turbulence_intensities=ti)
        else:
            ti = None

        # 3) Apply conditions to FlorisModel
        fm.set(
            wind_speeds=ws,
            wind_directions=wd,
            # same yaw for all turbines; change to list of different yaw
            # values if you want per-turbine actions later
            yaw_angles=[yaw] * num_t
        )

        # 4) Run FLORIS
        fm.run()

        # 5) Collect power (farm-level total)
        power = float(np.sum(fm.get_turbine_powers()))

        row = {
            **params,
            "wind_speed": ws,
            "wind_direction": wd,
            "yaw": yaw,
            "turbulence_intensity": ti,
            "num_turbines": num_t,
            "power": power,
        }
        rows.append(row)

    # 6) Save to CSV
    df = pd.DataFrame(rows)
    out_path = f"generated_data/{model_name}_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to: {out_path}")

    return df

# ENTRYPOINT

if __name__ == "__main__":
    env = load_environment(ENV_FILE)
    run_sweep(MODEL_NAME, N_SAMPLES, env)
