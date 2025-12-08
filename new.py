import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from floris import FlorisModel
import floris.flow_visualization as flowviz

# --- 0: CONFIGURE ---
# NREL 5MW diameter is approximately 126m
D = 126.0
N_TURBINES = 4
selected_wake_model = "gch.yaml" # cc.yaml, jensen,yaml, gch.yaml

# Define the 2x2 layout with 6D and 3D spacing
x_layout = [0, 0, 6 * D, 6 * D]  # Turbines at x=0 and x=756
y_layout = [0, 3 * D, 0, 3 * D]  # Turbines at y=0 and y=378

# --- 1: LOAD MODEL, SET LAYOUT ---
# Assuming 'farm_types/gch.yaml' is correctly accessible
fmodel = FlorisModel(f"farm_types/{selected_wake_model}")
fmodel.set(layout_x=x_layout, layout_y=y_layout)

print("Layout is set!")
print(f"Farm Layout: {N_TURBINES} turbines at {x_layout}, {y_layout}")
print("-" * 10)


# --- 2: WIND CONDITION PARAMETER SWEEP ---
wd_range = np.linspace(270.0, 280.0, 5)  # 5 directions
ws_range = np.array([8.0, 9.0, 10.0, 11.0]) # 4 wind speeds
ti_range = np.linspace(0.05, 0.20, 5)    # 5 turbulence intensity values

# 100 combinations of conditions
WD_grid, WS_grid, TI_grid = np.meshgrid(wd_range, ws_range, ti_range, indexing='ij')

# Flatten the grids for FLORIS's parallel calculation
all_wds = WD_grid.flatten()
all_wss = WS_grid.flatten()
all_tis = TI_grid.flatten()
N_findex = len(all_wds) # Should be 100

# Set the Wind Conditions once for the entire data generation process
fmodel.set(
    wind_directions=all_wds,
    wind_speeds=all_wss,
    turbulence_intensities=all_tis,
)

# --- 3: YAW ANGLE SETS (N_YawSets = 10 Unique Yaw Combinations) ---
# Each row defines a distinct yaw control strategy for the 4 turbines (T0, T1, T2, T3)
yaw_sets = np.array([
    [0.0, 0.0, 0.0, 0.0],      # Set 0: Baseline (No Yaw)
    [25.0, 0.0, 0.0, 0.0],     # Set 1: Yaw only T0 (Upstream)
    [-25.0, 0.0, 0.0, 0.0],    # Set 2: Yaw T0 negatively
    [20.0, 20.0, 0.0, 0.0],    # Set 3: Yaw T0 and T1
    [0.0, 25.0, 0.0, 0.0],     # Set 4: Yaw only T1
    [15.0, 15.0, 15.0, 15.0],  # Set 5: Uniform Yaw
    [20.0, -10.0, 0.0, 0.0],   # Set 6: Complex Yaw T0/T1
    [0.0, 25.0, 15.0, 0.0],    # Set 7: Sequential Yaw (T1/T2)
    [25.0, 25.0, 25.0, 25.0],  # Set 8: Max Uniform Yaw
    [10.0, 0.0, -10.0, 0.0],   # Set 9: T0/T2 Opposing Yaw
])
N_YAW_SETS = len(yaw_sets)

data_rows = []
print(f"Generating data: {N_findex} conditions * {N_YAW_SETS} yaw sets = {N_findex * N_YAW_SETS} total rows.")
print("-" * 10)


# --- 4: RUN FLORIS ON DIFF CONDITIONS (LOOP) ---
for yaw_index, single_yaw in enumerate(yaw_sets):
    # a) Broadcast the 1D yaw set to a (N_findex, N_turbines) array
    yaw_angles_array = np.tile(single_yaw, (N_findex, 1))

    # b) Run the simulation
    fmodel.set(yaw_angles=yaw_angles_array)
    fmodel.run()

    # c) Extract the power data
    power_kW = fmodel.get_turbine_powers() / 1000.0 # Shape (N_findex, N_turbines)

    # d) Build data rows
    for i in range(N_findex):
        row = {
            'Yaw_Set_ID': yaw_index,
            'WD': all_wds[i],
            'WS': all_wss[i],
            'TI': all_tis[i],
            'Farm_Power_kW': np.sum(power_kW[i, :]),
        }

        # Add individual turbine data
        for j in range(N_TURBINES):
            row[f'Yaw_T{j}'] = yaw_angles_array[i, j]
            row[f'Power_T{j}'] = power_kW[i, j]

        data_rows.append(row)

# Convert all rows to a single DataFrame
df_data = pd.DataFrame(data_rows)
print("Data generation complete.")
print(f"Final DataFrame Shape: {df_data.shape}")
df_data.to_csv(f"output/data/{selected_wake_model}_{N_TURBINES}turbines.csv", index=False)
print("Saved dataset to data folder.")
print("-" * 10)

# 5: WAKE VISUALIZATIONS (Optimized with calculate_horizontal_plane) ---
# NOTE: This code relies on the FModel having the 'calculate_horizontal_plane' method
# provided in your source code.

plot_indices = np.arange(0, 1000, 100)
H = 90.0
resolution = 100 # Use 100x100 resolution for speed

print("Generating Wake Visualizations...")

for plot_num, df_index in enumerate(plot_indices):
    row = df_data.iloc[df_index]
    
    # Extract parameters
    yaw_angles_to_plot = np.array([row[f'Yaw_T{j}'] for j in range(N_TURBINES)])
    wd_plot = row['WD']
    ws_plot = row['WS']
    ti_plot = row['TI']
    
    # --- 1. SET MODEL FOR VISUALIZATION (Equivalent to fmodel.set_for_viz) ---
    # Since we can't call fmodel.calculate_horizontal_plane directly on 'fmodel'
    # which has 100 wind conditions, we must replicate the setup process
    
    # Temporarily set the model for this single condition
    fmodel.set(
        wind_directions=[wd_plot],
        wind_speeds=[ws_plot],
        turbulence_intensities=[ti_plot],
        yaw_angles=np.array([yaw_angles_to_plot]) 
    )

    # --- 2. GENERATE THE CUTPLANE ---
    horizontal_cut_plane = fmodel.calculate_horizontal_plane(
        height=H,
        x_resolution=resolution,
        y_resolution=resolution,
        findex_for_viz=0 # Always 0 since we set the model to 1 condition
    )

    # --- 3. PLOT AND SAVE ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    flowviz.visualize_cut_plane(
        cut_plane=horizontal_cut_plane,
        ax=ax,
        title=(
            f"WAKE VISUALIZATION {plot_num+1}/10 | Index {df_index}\n"
            f"WD={wd_plot:.1f}°, WS={ws_plot:.1f} m/s, TI={ti_plot:.2f}\n"
            f"Yaw: {np.array2string(yaw_angles_to_plot, precision=1, separator=', ')}"
        )
    )
    
    # Save the figure
    plt.savefig(f"output/visualizations/wake_index_{df_index}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved wake plot {plot_num+1}/10.")

# Reset the model back to the 100 conditions for the diagnostic plots
fmodel.set(
    wind_directions=all_wds,
    wind_speeds=all_wss,
    turbulence_intensities=all_tis,
    yaw_angles=np.zeros((fmodel.n_findex, fmodel.core.farm.n_turbines))
)
print("-" * 10)

# --- 6: DIAGNOSTIC PLOTS ---
print("\nGenerating Diagnostic Plots...")

# 6.1 Plot: Farm Power vs. Wind Direction (Sweep Plot)
target_ti = 0.125
target_ws = 9.0

df_filtered = df_data[
    (df_data['TI'] == target_ti) & 
    (df_data['WS'] == target_ws)
].sort_values(by='WD')

df_baseline = df_filtered[df_filtered['Yaw_Set_ID'] == 0]
df_steer = df_filtered[df_filtered['Yaw_Set_ID'] == 1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_baseline['WD'], df_baseline['Farm_Power_kW'], 'ko-', label='Baseline (0 deg Yaw)')
ax.plot(df_steer['WD'], df_steer['Farm_Power_kW'], 'r^--', label='Steering (T0 Yaw = 25 deg)')

ax.set_title(f'Farm Power vs. Wind Direction (WS={target_ws} m/s, TI={target_ti})')
ax.set_xlabel('Wind Direction (degrees)')
ax.set_ylabel('Farm Power (kW)')
ax.legend()
ax.grid(True, linestyle='--')

plt.savefig(os.path.join("output/plots", "diag_power_vs_wd_sweep.png"), bbox_inches='tight')
plt.close(fig)
print("  Saved Diagnostic Plot 1/3 (Power vs WD).")


# 6.2 Plot: Power Gain vs. Turbulence Intensity
df_baseline_avg = df_data[df_data['Yaw_Set_ID'] == 0].groupby('TI')['Farm_Power_kW'].mean().reset_index()
df_baseline_avg.rename(columns={'Farm_Power_kW': 'Power_Baseline_Avg'}, inplace=True)

df_plot = df_data[df_data['Yaw_Set_ID'] == 1]
df_plot = pd.merge(df_plot, df_baseline_avg, on='TI', how='left')

df_plot['Power_Gain_%'] = 100 * (df_plot['Farm_Power_kW'] - df_plot['Power_Baseline_Avg']) / df_plot['Power_Baseline_Avg']
df_ti_gain = df_plot.groupby('TI')['Power_Gain_%'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_ti_gain['TI'], df_ti_gain['Power_Gain_%'], 'b^-', label='Average Power Gain from Steering')

ax.set_title('Effect of Turbulence Intensity on Wake Steering Power Gain')
ax.set_xlabel('Turbulence Intensity (TI)')
ax.set_ylabel('Power Gain (%)')
ax.grid(True, linestyle='--')

plt.savefig(os.path.join("output/plots", "diag_gain_vs_ti.png"), bbox_inches='tight')
plt.close(fig)
print("  Saved Diagnostic Plot 2/3 (Gain vs TI).")


# 6.3 Plot: Box Plot of Farm Power by Yaw Set
fig, ax = plt.subplots(figsize=(12, 6))
df_data.boxplot(column='Farm_Power_kW', by='Yaw_Set_ID', grid=True, ax=ax)

ax.set_title('Distribution of Farm Power by Yaw Angle Set')
ax.set_xlabel('Yaw Angle Set ID')
ax.set_ylabel('Farm Power (kW)')
fig.suptitle('') # Suppress auto suptitle

plt.savefig(os.path.join("output/plots", "diag_boxplot_power_by_yaw_set.png"), bbox_inches='tight')
plt.close(fig)
print("  Saved Diagnostic Plot 3/3 (Box Plot).")
print("-" * 30)
print("All data and plots have been successfully generated and saved.")