import cdsapi
import numpy as np

variables = [
    "10m_u_component_of_wind",  # ! 0 u10
    "10m_v_component_of_wind",  # ! 0 v10
    "mean_sea_level_pressure",  # ! 1 msl
]

c = cdsapi.Client()

for i in range(0, len(variables), 2):
    current_item = variables[i]
    if i + 1 < len(variables):
        vars = [variables[i], variables[i + 1]]
    else:
        vars = [variables[i]]
    print(vars, "p", int(np.ceil(i / 2)))
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": vars,
            "year": list(range(2022, 2024)),
            "month": list(range(1, 13)),
            "day": list(range(1, 32)),
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "area": [
                -40,
                108,
                -22,
                128,
            ],  # Replace with your desired bounding box coordinates
            "format": "netcdf",
        },
        f"/YOUR_DESIRED_DIRECTORY_{i}.nc"
        )