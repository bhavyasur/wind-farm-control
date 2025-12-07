import floris
import matplotlib.pyplot as plt
from floris import FlorisModel
fmodel = FlorisModel("model name here") # model is instantiated, can be used to inspect data
x, y = fmodel.get_turbine_layout()

num_turbines = 3
x_2x2 = [0, 400, 0, 800]
y_2x2 = [0, 0, 800, 800]
fmodel.set(layout_x=x_2x2, layout_y=y_2x2)



turbine_type = "nrel_5MW"
