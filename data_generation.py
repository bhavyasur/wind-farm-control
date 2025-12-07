import numpy as np
import matplotlib.pyplot as plt

from floris import FlorisModel, TimeSeries
import floris.flow_visualization as flowviz
import floris.layout_visualization as layoutviz

fmodel = FlorisModel("farm_types/gch.yaml") # model is instantiated, can be used to inspect data
x, y = fmodel.get_turbine_layout()
print("initial turbine layout")
print("     x       y")
for _x, _y in zip(x, y):
    print(f"{_x:6.1f}, {_y:6.1f}")

num_turbines = 3
x_2x2 = [0, 400, 0, 800]
y_2x2 = [0, 0, 800, 800]
fmodel.set(layout_x=x_2x2, layout_y=y_2x2)



turbine_type = "nrel_5MW"
