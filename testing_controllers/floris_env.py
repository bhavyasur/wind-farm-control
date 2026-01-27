import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from floris import FlorisModel

class FlorisEnv(gym.Env):
    def __init__(self):
        super(FlorisEnv, self).__init__()
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        config_path = project_root / "data_generation" / "farm_types" / "gch.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"FLORIS config not found at: {config_path}")

        # FLORIS SETUP
        self.fmodel = FlorisModel(str(config_path))
        
        D = 126.0
        self.x_layout = [0, 0, 6 * D, 6 * D]
        self.y_layout = [0, 3 * D, 0, 3 * D]
        self.fmodel.set(layout_x=self.x_layout, layout_y=self.y_layout)
        
        self.n_turbines = len(self.x_layout)
        
        # GYMNASIUM ACTION SPACE
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_turbines,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([270.0, 8.0, 0.05]), 
            high=np.array([280.0, 11.0, 0.20]), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # sweep ranges
        self.current_wd = np.random.uniform(270.0, 280.0)
        self.current_ws = np.random.uniform(8.0, 11.0)
        self.current_ti = np.random.uniform(0.05, 0.20)
        
        obs = np.array([self.current_wd, self.current_ws, self.current_ti], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # scaling (-1 to 1) to (-25 to 25) degrees
        yaw_angles = action * 25.0
        
        # BASELINE CALCULATION (yaw = 0)
        self.fmodel.set(
            wind_directions=np.array([self.current_wd]),
            wind_speeds=np.array([self.current_ws]),
            turbulence_intensities=np.array([self.current_ti]),
            yaw_angles=np.zeros((1, self.n_turbines)) # Force zero yaw
        )
        self.fmodel.run()
        baseline_power_mw = np.sum(self.fmodel.get_turbine_powers()) / 1e6

        # PERFORMANCE (POWER)
        self.fmodel.set(yaw_angles=np.array([yaw_angles]))
        self.fmodel.run()
        rl_power_mw = np.sum(self.fmodel.get_turbine_powers()) / 1e6
        
        # GAIN
        # Positive means the RL agent is helping; negative means it's hurting
        gain_pct = 100 * (rl_power_mw - baseline_power_mw) / baseline_power_mw
        
        
        terminated = True 
        truncated = False
        
        # Current Wind State is the Observation
        obs = np.array([self.current_wd, self.current_ws, self.current_ti], dtype=np.float32)
        
        # The 'info' dict is visible in logs and your test script
        info = {
            "power_kw": rl_power_mw * 1000,
            "baseline_kw": baseline_power_mw * 1000,
            "gain_pct": gain_pct
        }
        
        return obs, rl_power_mw, terminated, truncated, info
    
if __name__ == "__main__":
    env = FlorisEnv()
    obs, _ = env.reset()
    
    # random step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("-" * 30)
    print(f"WIND: {obs[0]:.1f}° at {obs[1]:.1f} m/s")
    print(f"YAW DEGREES: {action * 25.0}")
    print(f"BASELINE:    {info['baseline_kw']:.2f} kW")
    print(f"RL AGENT:    {info['power_kw']:.2f} kW")
    print(f"GAIN:        {info['gain_pct']:.2f}%")
    print("-" * 30)