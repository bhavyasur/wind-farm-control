import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

from models.jensen import JensenWakeModel
from models.gauss import GaussWakeModel
from models.curl import CurlWakeModel
from models.turboparkgauss import TurboParkGaussWakeModel


# CONFIG

MODEL_NAME = "jensen"
N_SAMPLES  = 1000
ENV_FILE   = "configs/env.yaml"

model_map = {
    "jensen": JensenWakeModel,
    "gauss":  GaussWakeModel,
    "curl":   CurlWakeModel,
    "turboparkgauss": TurboParkGaussWakeModel
}


# HELPERS

def load_environment(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def sample_env_value(entry: dict) -> float:
    if "fixed" in entry:
        return float(entry["fixed"])
    return float(np.random.uniform(entry["min"], entry["max"]))


def generate_layout(num_turbines: int, spacing: float = 500.0):
    layout_x = [i * spacing for i in range(num_turbines)]
    layout_y = [0.0] * num_turbines
    return layout_x, layout_y


# PLOT CODE

def plot_samples(samples, model, output_image_path):
    """
    FLORIS 4.5.1 conda-build:
    plane.df has columns ['x1', 'x2', 'x3', 'u', 'v', 'w']
    plane.resolution gives (nx, ny)
    """

    fm = model.fm

    n = len(samples)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for ax_idx, (ax, s) in enumerate(zip(axes, samples)):
        idx = s["index"]

        # restores FLORIS layout
        fm.set(
            layout_x=s["layout_x"],
            layout_y=s["layout_y"],
            wind_speeds=[s["wind_speed"]],
            wind_directions=[s["wind_direction"]],
            yaw_angles=[[s["yaw"]] * len(s["layout_x"])],
            turbulence_intensities=[s["turbulence_intensity"]],
        )
        model.apply_params(s["params"])

        # run model
        fm.run()

        plane = fm.calculate_horizontal_plane(height=90.0)

        df_plane = plane.df
        nx, ny = plane.resolution

        X = df_plane["x1"].values.reshape(nx, ny)
        Y = df_plane["x2"].values.reshape(nx, ny)
        U = df_plane["u"].values.reshape(nx, ny)

        c = ax.contourf(X, Y, U, levels=40, cmap="viridis")
        ax.set_title(
            f"Sample {idx+1}\n"
            f"WS={s['wind_speed']:.1f}, WD={s['wind_direction']:.1f}, yaw={s['yaw']:.1f}",
            fontsize=8
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        fig.colorbar(c, ax=ax, shrink=0.75)

    for j in range(ax_idx + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Wake Field Snapshots at Selected Samples", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=150)
    print(f"Saved visualization image to: {output_image_path}")

    plt.show()


# MAIN SWEEP

def run_sweep(model_name: str, N: int, env: dict):
    ModelClass = model_map[model_name]
    model = ModelClass()
    fm = model.fm

    # Create turbine layout
    num_t = int(sample_env_value(env["num_turbines"]))
    layout_x, layout_y = generate_layout(num_t)
    fm.set(layout_x=layout_x, layout_y=layout_y)

    # Visualization indices
    vis_indices = sorted(set([0] + list(range(99, N, 100))))
    vis_samples = []

    rows = []

    for i in range(N):
        params = model.sample_params()
        model.apply_params(params)

        ws  = sample_env_value(env["wind_speed"])
        wd  = sample_env_value(env["wind_direction"])
        yaw = sample_env_value(env["yaw"])

        if "turbulence_intensity" in env:
            ti = sample_env_value(env["turbulence_intensity"])
            fm.set(turbulence_intensities=[ti])
        else:
            ti = None

        fm.set(
            wind_speeds=[ws],
            wind_directions=[wd],
            yaw_angles=[[yaw] * num_t],
        )

        fm.run()

        power = float(np.sum(fm.get_turbine_powers()))

        rows.append({
            **params,
            "wind_speed": ws,
            "wind_direction": wd,
            "yaw": yaw,
            "turbulence_intensity": ti,
            "num_turbines": num_t,
            "power": power,
        })

        if i in vis_indices:
            vis_samples.append({
                "index": i,
                "params": params.copy(),
                "wind_speed": ws,
                "wind_direction": wd,
                "yaw": yaw,
                "turbulence_intensity": ti,
                "layout_x": layout_x.copy(),
                "layout_y": layout_y.copy(),
            })

    out_path = f"generated_data/{model_name}_{num_t}turbines_data.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to: {out_path}")

    image_path = out_path.replace(".csv", "_image.png")
    plot_samples(vis_samples, model, image_path)

    return df


# ENTRYPOINT

if __name__ == "__main__":
    env = load_environment(ENV_FILE)
    run_sweep(MODEL_NAME, N_SAMPLES, env)
