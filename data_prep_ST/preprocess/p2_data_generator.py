import xarray as xr
import numpy as np
import calendar
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_prep_ST.preprocess import preprocess
import mlflow
import rasterio


class Train_Test:
    # ------------------------------------------------------------------------------------#
    #       This class is responsible for                                                 #
    #       1. merging dpird and ecmwf data if required                                   #
    #       2. adding time features (not popularizing spatilly yet)                       #
    #       3. splitting training and testing set                                         #
    #       4. scaling training and applying the scale on testing                         #
    #       5. store the training and testing set                                         #
    #       Note: the data_vars include dpird+ecmwf+tvars, and 'cumulative_month'         #
    # ------------------------------------------------------------------------------------#
    def __init__(self, args):
        self.args = args
        self.tvars = [
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]

    def _customise_grid(self, ds):
        dataset = ds.sel(
            latitude=slice(self.args.lat_min, self.args.lat_max + self.args.grid_deg),
            longitude=slice(self.args.lon_min, self.args.lon_max + self.args.grid_deg),
        )
        return dataset

    def _read_rawdata(self):
        if self.args.datasrc == 0:
            ds = xr.open_dataset(self.args.dpird_dest_path)
            ds = ds.sel(time=slice(self.args.start_dpird, self.args.end))
            ds = ds[self.args.vars_dpird]
        elif self.args.datasrc == 1:
            ds = xr.open_dataset(self.args.cds_dest_path)
            ds = ds.sel(time=slice(self.args.start_dpird, self.args.end))
            ds = ds[self.args.vars_cds]
        else:
            ds_dpird = xr.open_dataset(self.args.dpird_dest_path)
            ds_dpird = ds_dpird.sel(time=slice(self.args.start_dpird, self.args.end))
            ds_dpird = ds_dpird[self.args.vars_dpird]
            ds_ecmwf = xr.open_dataset(self.args.cds_dest_path)
            ds_ecmwf = ds_ecmwf.sel(time=slice(self.args.start_dpird, self.args.end))
            ds_ecmwf = ds_ecmwf[self.args.vars_cds]
            ds = xr.merge([ds_dpird, ds_ecmwf])

        # add terrain
        if self.args.vars_terrain:
            ds_terrain = xr.open_dataset(self.args.terrain_dest_path)
            ds = xr.merge([ds, ds_terrain])

        if self.args.customise_grid:
            ds = self._customise_grid(ds)
        return ds

    def _add_cumulative_month(self, ds):
        if "cumulative_month" not in ds.data_vars.keys():
            ds["cumulative_month"] = (
                (ds["time.year"] - ds["time.year"][0]) * 12 + ds["time.month"] - 1
            )
        return ds

    def _get_days_in_year(self, year):
        if calendar.isleap(year):
            return 366  # Leap year has 366 days
        else:
            return 365  # Common year has 365 days

    def _add_time_features(self, ds):
        ds["month_sin"] = (("time"), np.sin(2 * np.pi * ds["time.month"].data / 12))
        ds["month_cos"] = (("time"), np.cos(2 * np.pi * ds["time.month"].data / 12))

        hour = np.array(ds["time"].dt.hour + ds["time"].dt.minute / 60)
        # Calculate sin and cos of the hour in the day
        ds["hour_sin"] = (("time"), np.sin(2 * np.pi * hour / 24))
        ds["hour_cos"] = (("time"), np.cos(2 * np.pi * hour / 24))

        # Calculate sin and cos of the day in the year
        day_in_year = np.array(ds["time"].dt.dayofyear)
        leap_years = np.array([calendar.isleap(year) for year in ds["time.year"]])
        days_in_year = np.where(leap_years, 366, 365)
        ds["day_sin"] = (("time"), np.sin(2 * np.pi * day_in_year / days_in_year))
        ds["day_cos"] = (("time"), np.cos(2 * np.pi * day_in_year / days_in_year))

        return ds

    # split train and test data
    def train_test_dataset(self) -> tuple[xr.Dataset, xr.Dataset]:
        ds = self._read_rawdata()
        ds = self._add_cumulative_month(ds)
        ds = self._add_time_features(ds)
        test_n_samples = self.args.test_n_days * 4 * 24

        train_months, test_months = [], []
        for month in np.arange(ds["cumulative_month"].max().item() + 1):
            dataset_month = ds.where(ds["cumulative_month"] == month, drop=True)
            test_month = dataset_month.isel(time=slice(-test_n_samples, None))
            test_months.append(test_month)
            train_month = dataset_month.isel(time=slice(None, -test_n_samples))
            train_months.append(train_month)
        train_ds = xr.concat(train_months, dim="time")
        test_ds = xr.concat(test_months, dim="time")
        return train_ds, test_ds

    def scale_x_dpird(
        self, train_ds: xr.Dataset, test_ds: xr.Dataset
    ) -> tuple[xr.Dataset, xr.Dataset]:
        if self.args.datasrc == 0:
            vars = self.args.vars_dpird
        elif self.args.datasrc == 1:
            vars = self.args.vars_cds
        else:
            vars = self.args.vars_dpird + self.args.vars_cds
        if self.args.vars_terrain:
            vars = vars + ["terrain"]
        # the dpird datasets allows for nans, because it is a sparse dataset
        # create arrays
        train_4darray = np.array([train_ds[var].as_numpy() for var in vars])
        test_4darray = np.array([test_ds[var].as_numpy() for var in vars])

        # mask nan values in the 4d array
        masked_train_4darray = np.ma.masked_invalid(train_4darray)
        masked_test_4darray = np.ma.masked_invalid(test_4darray)

        # reshape
        shape4d_train = masked_train_4darray.shape
        shape4d_test = masked_test_4darray.shape

        train_2darray = train_4darray.reshape(-1, len(vars))
        test_2darray = test_4darray.reshape(-1, len(vars))

        scaler = StandardScaler()
        std_train_2darray = scaler.fit_transform(train_2darray)
        std_train_4darray = std_train_2darray.reshape(shape4d_train)
        std_test_2darray = scaler.transform(test_2darray)
        std_test_4darray = std_test_2darray.reshape(shape4d_test)

        # Fill NaN values with 0 in the standardized 4D array
        std_train_4darray = np.where(masked_train_4darray.mask, 0, std_train_4darray)
        std_test_4darray = np.where(masked_test_4darray.mask, 0, std_test_4darray)

        for idx, var in enumerate(vars):
            train_ds[var] = (("time", "latitude", "longitude"), std_train_4darray[idx])
            test_ds[var] = (("time", "latitude", "longitude"), std_test_4darray[idx])

        return train_ds, test_ds

    # check the train_ds and test_ds should contain 2 scaled dpird features and cds features, and 6 scaled time features, and 1 cumulative month

    def main_generate_dataset(self) -> tuple[xr.Dataset, xr.Dataset]:
        if (
            os.path.isfile(self.args.train_path)
            and os.path.isfile(self.args.test_path)
            and os.path.isfile(self.args.test_noscale_path)
        ):
            print("The training and testing data is available, loading...")
            train = xr.open_dataset(self.args.train_path)
            test = xr.open_dataset(self.args.test_path)
            print("The data was loaded")
            return train, test
        else:
            print(
                "The training and testing data is not available, creating and storing..."
            )
            dir = os.path.dirname(self.args.train_path)
            if not os.path.exists(dir):
                raise ValueError(f"Pls mk dir {dir}")

            train, test = self.train_test_dataset()
            if not os.path.exists(self.args.test_noscale_path):
                test.to_netcdf(self.args.test_noscale_path)
            print("test_noscale is done")
            train, test = self.scale_x_dpird(train, test)

            train.to_netcdf(self.args.train_path)
            print("train is done")
            test.to_netcdf(self.args.test_path)
            print("test is done")
            print("The data was loaded")
            # return train, test


class STDataset(Dataset):
    def __init__(self, args, flag: str = "train", mode: str = "all"):
        self.args = args

        assert flag in ["train", "test_star", "test_grid"]
        assert mode in [
            "all",
            "fnt",
        ]  # all: feature and timefeature are combined into one set, fnt: feature and timefeature are output individually
        self.mode = mode

        self.flag = flag
        if self.flag == "train":
            ds_path = self.args.train_path
            assert os.path.isfile(ds_path)
        else:  # for flag is test_star or test_grid
            ds_path = self.args.test_path
            ds_noscale_path = self.args.test_noscale_path
            self.ds_noscale = xr.open_dataset(ds_noscale_path)
            assert os.path.isfile(ds_path)
            assert os.path.isfile(ds_noscale_path)

        self.ds = xr.open_dataset(ds_path)
        self.ds_y = xr.open_dataset(self.args.label_dest_path)
        self.ds_y3m = xr.open_dataset(self.args.label3m_dest_path)

        if self.flag == "test_grid":
            self.ds = self.ds.sel(
                time=slice(self.args.test_grid_start, self.args.test_grid_end)
            )
            self.ds_y = self.ds_y.sel(
                time=slice(self.args.test_grid_start, self.args.test_grid_end)
            )

            self.ds_y3m = self.ds_y3m.sel(
                time=slice(self.args.test_grid_start, self.args.test_grid_end)
            )

        if self.args.customise_grid:
            self.ds_y = self._customise_grid(self.ds_y)

            self.ds_y3m = self._customise_grid(self.ds_y3m)

        self.tvars = [
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]
        if self.args.datasrc == 0:
            self.vars = self.args.vars_dpird
        elif self.args.datasrc == 1:
            self.vars = self.args.vars_cds
        else:
            self.vars = self.args.vars_cds + self.args.vars_dpird
        if self.args.vars_terrain:
            self.vars = self.vars + ["terrain"]
        self.start = pd.to_datetime(self.args.start_dpird)
        # T_hr: the duration of each sample, i.e. 3d,2d,1d,12hr
        # L_hr: the moving window between x and y.
        # S_min the sliding window of the next sample
        # F_hr: the extending window for ECMWF data
        self.I_min = 15  # the interval of each stamp
        self.ix_time = self._match_idx()
        self._create_label_idx()

        self._create_label3m_idx()

    def __len__(self):
        # total number of training or testing samples
        n_stmp_per_xy_cds = (self.args.T_hr + self.args.F_hr + self.args.L_hr) * 4
        n_stmp_per_xy_dpird = (self.args.T_hr + self.args.L_hr) * 4
        length = 0
        for mon in range(
            int(self.ds["cumulative_month"].min().item()),
            int(self.ds["cumulative_month"].max().item()) + 1,
        ):
            ds_mon = self.ds.where(self.ds["cumulative_month"] == mon, drop=True)
            length = length + np.floor(
                (len(ds_mon["time"]) - n_stmp_per_xy_cds)
                / (self.args.S_min / self.I_min)
            )
            length = int(length)
        return length

    def _customise_grid(self, ds):
        dataset = ds.sel(
            latitude=slice(self.args.lat_min, self.args.lat_max + self.args.grid_deg),
            longitude=slice(self.args.lon_min, self.args.lon_max + self.args.grid_deg),
        )
        return dataset

    def _create_label_idx(self):
        # create dataframe where locate the ix of lat and lon for labels

        grid_label = xr.load_dataset(self.args.label_dest_path)

        if self.args.customise_grid:
            grid_label = self._customise_grid(grid_label)
        star_loc = preprocess.create_label_ix(
            self.args.stars_coords, grid_label, self.args
        )
        star_loc.to_csv(self.args.star_coord_ix, index=False)
        if self.args.mlflow:
            mlflow.log_artifact(self.args.star_coord_ix)

    def _create_label3m_idx(self):
        # create dataframe where locate the ix of lat and lon for labels

        grid_label3m = xr.load_dataset(self.args.label3m_dest_path)

        if self.args.customise_grid:
            grid_label = self._customise_grid(grid_label3m)
        station3m_loc = preprocess.create_label3m_ix(
            self.args.stations_coords, grid_label3m, self.args
        )
        station3m_loc.to_csv(self.args.station3m_coord_ix, index=False)
        if self.args.mlflow:
            mlflow.log_artifact(self.args.station3m_coord_ix)

    def _match_idx(self):
        # map idx with the index of the samples
        # NOTICE: the samples are not strictly continuous on time.
        # The break of the end of month should be taken care of.
        ix_time = []

        n_stmp_per_xy_cds = (self.args.T_hr + self.args.F_hr + self.args.L_hr) * 4
        n_stmp_per_xy_dpird = (self.args.T_hr + self.args.L_hr) * 4

        for mon in range(
            int(self.ds["cumulative_month"].min().item()),
            int(self.ds["cumulative_month"].max().item()) + 1,
        ):
            ds_mon = self.ds.where(self.ds["cumulative_month"] == mon, drop=True)
            last_start_time = ds_mon["time"][
                len(ds_mon["time"]) - n_stmp_per_xy_cds
            ].values
            start_time = pd.to_datetime(ds_mon["time"][0].values)
            while start_time <= last_start_time:
                ix_time.append(start_time)
                start_time = start_time + pd.to_timedelta(self.args.S_min, unit="m")

        return ix_time

    def _get_time(self, idx):
        x_start_time = self.ix_time[idx]
        y_start_time = x_start_time + pd.to_timedelta(self.args.L_hr, unit="h")

        if self.args.datasrc == 0:
            x_end_time = (
                x_start_time
                + pd.to_timedelta(self.args.T_hr, unit="h")
                - pd.to_timedelta(self.I_min, unit="m")
            )  # [n,T,lat,lon]
            y_end_time = (
                y_start_time
                + pd.to_timedelta(self.args.T_hr, unit="h")
                - pd.to_timedelta(self.I_min, unit="m")
            )  # [n,T,lat,lon]
        else:
            x_end_time = (
                x_start_time
                + pd.to_timedelta(self.args.T_hr, unit="h")
                + pd.to_timedelta(self.args.F_hr, unit="h")
                - pd.to_timedelta(self.I_min, unit="m")
            )  # [n,T+F,lat,lon]
            y_end_time = (
                y_start_time
                + pd.to_timedelta(self.args.T_hr, unit="h")
                + pd.to_timedelta(self.args.F_hr, unit="h")
                - pd.to_timedelta(self.I_min, unit="m")
            )  # [n,T+F,lat,lon]

        return x_start_time, x_end_time, y_start_time, y_end_time

    def _getitem_all(self, idx):  # need to modify for dpird and cds set
        # popularize the time features for all lat and lon
        for var in self.tvars:
            self.ds[var] = xr.broadcast(self.ds[var], self.ds["latitude"])[0]
            self.ds[var] = xr.broadcast(self.ds[var], self.ds["longitude"])[0]

        x_start, x_end, y_start, y_end = self._get_time(idx)
        x_ds = self.ds.sel(time=slice(x_start, x_end))
        if self.flag in ["test_star", "test_grid"]:
            x_noscale_ds = self.ds_noscale.sel(time=slice(x_start, x_end))
            x_noscale_ds = x_noscale_ds[["u10", "v10"]]
            x_ecmwf = torch.tensor(
                np.array([x_noscale_ds[var].values for var in ["u10", "v10"]]),
                dtype=torch.float32,
            )
        y_ds = self.ds_y.sel(time=slice(y_start, y_end))

        y3m_ds = self.ds_y3m.sel(time=slice(y_start, y_end))
        if self.args.datasrc == 2:
            x_T = x_start + pd.to_timedelta(self.args.T_hr, unit="h")
            x_ds[self.args.vars_dpird].loc[dict(time=slice(x_T, x_end))] = (
                0  #!!! This makes sure that we only use ecmwf forecast data from T to T+F
            )

        x_4d_ts = torch.tensor(
            np.array([x_ds[var].values for var in self.vars + self.tvars]),
            dtype=torch.float32,
        )  # [n_var,T,lat,lon]
        y_4d_ts = torch.tensor(
            np.array([y_ds[var].values for var in self.args.labels[:2]]),
            dtype=torch.float32,
        )  # [n_var,T,lat,lon]

        y3m_4d_ts = torch.tensor(
            np.array([y3m_ds[var].values for var in ["wind_3m_u", "wind_3m_v"]]),
            dtype=torch.float32,
        )  # [n_var,T,lat,lon]
        y_time = torch.tensor(y_ds["time"].astype(np.float64).values)

        # mask the nans
        # y_4dm_ts = torch.where(torch.isnan(y_4d_ts), torch.zeros_like(y_4d_ts), y_4d_ts)
        if self.flag == "train":
            return x_4d_ts, y_4d_ts, y_time, y3m_4d_ts
        elif self.flag in ["test_star", "test_grid"]:
            return x_4d_ts, y_4d_ts, y_time, x_ecmwf, y3m_4d_ts

    def _getitem_fnt(self, idx):  # need to modify for dpird and cds set
        x_start, x_end, y_start, y_end = self._get_time(idx)
        x_ds = self.ds.sel(time=slice(x_start, x_end))
        y_ds = self.ds_y.sel(time=slice(y_start, y_end))
        if self.args.datasrc == 2:
            x_T = x_start + pd.to_timedelta(self.args.T_hr, unit="h")
            x_ds.loc[dict(time=slice(x_T, x_end)), self.args.vars_dpird] = 0

        x_4d_ts = torch.tensor(
            np.array([x_ds[var].values for var in self.vars]), dtype=torch.float32
        )  # [n_var,T,lat,lon]
        x_enc_ts = torch.tensor(
            np.array([x_ds[var].values for var in self.tvars]), dtype=torch.float32
        )  # [n_tvar,T]
        y_4d_ts = torch.tensor(
            np.array([y_ds[var].values for var in self.args.labels]),
            dtype=torch.float32,
        )  # [n_var,T,lat,lon]
        y_time = torch.tensor(y_ds["time"].astype(np.float64).values)
        # mask the nans
        # y_4dm_ts = torch.where(torch.isnan(y_4d_ts), torch.zeros_like(y_4d_ts), y_4d_ts)
        return x_4d_ts, x_enc_ts, y_4d_ts, y_time

    def __getitem__(self, idx):
        if self.mode == "all":
            return self._getitem_all(idx)
        else:
            return self._getitem_fnt(idx)

    def get_shape(self):
        x_4d_ts, *_ = self.__getitem__(0)

        n_feat, len_time, len_lat, len_lon = x_4d_ts.shape

        return n_feat, len_time, len_lat, len_lon


def data_provider(args, flag, mode="all", rank=None):
    if flag == "train":
        shuffle = True
        drop_last = False
        num_workers = 0 if args.distributed else 4
        batch_size = args.batch_size
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
        batch_size = 1

    dataset = STDataset(args, flag=flag, mode=mode)
    if args.distributed and args.flag == "train":
        sampler = DistributedSampler(
            dataset, rank=rank, shuffle=shuffle, drop_last=drop_last
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            pin_memory_device="cuda:{}".format(rank),
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    print(
        "dataloader provides with x in shape of [n_batch,n_feature,n_time,n_lat,n_lon]. \n "
        "If using dpird and ecmwf at the same time, ecmwf is extended for F_hr, \n"
        "while the data of dpird is set to 0 as unknown. \n "
        "If the labels are wind_10m_u/v, the loss function is calculated on the labelled stations. \n "
        "If the labels are wind_10m_u/v and wind_3m_u/v, the loss function is calculated on lablled staitons \n "
        "and the stations with 3m data."
    )
    return dataloader, sampler
