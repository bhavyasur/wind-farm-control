import floris
from floris import FlorisModel
fm = FlorisModel("configs/farm_setup.yaml")
print([m for m in dir(fm) if "plane" in m.lower()])

fm.run()
plane = fm.calculate_horizontal_plane(height=90.0)
print(sorted(dir(plane)))

print(plane.resolution)

print(plane.df.columns)
