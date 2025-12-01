import numpy as np
import pandas as pd
import yaml

# Import model classes
from models.jensen import JensenModel
from models.gauss import GaussModel
from models.gch import CurlModel
from models.curl import GCHModel


# CONFIGURATION: Set everything here
with open('configs/farm_setup.yml', 'r') as file:
    farm_setup = yaml.safe_load(file)

MODEL_NAME = "gch"         # "jensen", "gauss", "curl", or "gch"
N_SAMPLES = 1000           # number of rows of data to generate

# MODEL MAP (do not edit)

model_map = {
    "jensen": JensenModel,
    "gauss": GaussModel,
    "curl":  CurlModel,
    "gch":   GCHModel
}

# ------------------------------------
# MAIN SWEEP FUNCTION
# ------------------------------------

def run_sweep(model_name, N):
    ModelClass = model_map[model_name]
    model = ModelClass()
    fm = model.fm

    rows = []

    for _ in range(N):

        # ---- Sample wake model parameters ----
        params = model.sample_params()
        model.apply_params(params)

        # ---- Sample environment values ----
        ws = float(np.random.uniform(4, 14))        # wind speed
        wd = float(np.random.uniform(250, 290))     # wind direction
        yaw = float(np.random.uniform(-25, 25))     # yaw (deg)

        # ---- Set environment ----
        fm.set(
            wind_speeds=ws,
            wind_directions=wd,
            yaw_angles=[yaw]
        )

        fm.run()

        # ---- Extract output ----
        power = float(np.sum(fm.get_turbine_powers()))

        row = {
            **params,
            "wind_speed": ws,
            "wind_direction": wd,
            "yaw": yaw,
            "power": power
        }
        rows.append(row)

    # ---- Save to CSV ----
    output_path = f"output/{model_name}_data.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to: {output_path}")

    return df


# ------------------------------------
# EXECUTE WHEN RUN
# ------------------------------------

if __name__ == "__main__":
    run_sweep(MODEL_NAME, N_SAMPLES)
