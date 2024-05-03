from pykrige.ok import OrdinaryKriging
import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial import distance_matrix
from scipy.interpolate import Rbf


class Interpolation_model_grid:
    def __init__(
        self,
        data: xr.Dataset,
        grid_deg,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    ):
        self.data = data
        self.features = data.data_vars.keys()
        self.grid_deg = grid_deg
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.__make_rectangular_grid__()

    def __make_rectangular_grid__(self):
        self.rectgrid_lat = np.linspace(
            self.lat_min, self.lat_max, num=32, endpoint=False, dtype="float64"
        )
        self.rectgrid_lon = np.linspace(
            self.lon_min, self.lon_max, num=32, endpoint=False, dtype="float64"
        )
        print(self.rectgrid_lat.shape)
        print(self.rectgrid_lon.shape)

    def _check_unique(self, data_t_f):
        unique_values = np.unique(data_t_f)  # Get unique values, ignoring NaNs
        unique_values = unique_values[~np.isnan(unique_values)]  # Remove NaNs
        return unique_values

    def OrdinaryKriging_exp(self, data_t_f):
        # data_t_f is an array in xarray, with one variable, one time stamp, and latitude and longitude
        uni = self._check_unique(data_t_f)
        if len(uni) <= 1:
            filled_z = np.full((len(self.rectgrid_lat), len(self.rectgrid_lon)), uni)
        else:
            flat_t_f = data_t_f.stack(z=("latitude", "longitude"))
            OK_exp = OrdinaryKriging(
                x=flat_t_f["latitude"],
                y=flat_t_f["longitude"],
                z=flat_t_f.values,
                variogram_model="exponential",
                verbose=False,
                enable_plotting=False,
                coordinates_type="geographic",
            )
            z, ss = OK_exp.execute("grid", self.rectgrid_lon, self.rectgrid_lat)
            filled_z = np.ma.filled(z, fill_value=None)
        return filled_z

    def krig_3D_vars_xr(self):
        feature_data = {
            feature: np.zeros(
                (len(self.data["time"]), len(self.rectgrid_lat), len(self.rectgrid_lon))
            )
            for feature in self.features
        }
        dataset = xr.Dataset()
        for i, time_stamp in enumerate(self.data["time"]):
            print(time_stamp.values)
            for feature in self.features:
                print("processing", feature)
                data_t_f = self.data[feature].sel(time=time_stamp)
                array_values = self.OrdinaryKriging_exp(data_t_f)
                feature_data[feature][i, :, :] = array_values

        for feature in self.features:
            data_array = xr.DataArray(
                data=feature_data[feature],
                dims=("time", "latitude", "longitude"),
                coords={
                    "time": self.data["time"],
                    "latitude": self.rectgrid_lat,
                    "longitude": self.rectgrid_lon,
                },
                name=feature,
            )
            dataset[feature] = data_array
        return dataset

    def ln_interp_3D_vars_xr(self):
        ds = self.data[self.features].interp(
            latitude=self.rectgrid_lat,
            longitude=self.rectgrid_lon,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        # ds = ds.interpolate_na()
        return ds
