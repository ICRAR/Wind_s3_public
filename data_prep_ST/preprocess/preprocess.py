import pandas as pd
import math, os
import numpy as np
import xarray as xr
import datetime
from typing import List
from data_prep_ST.preprocess.interpolation_model import Interpolation_model_grid
from colorama import init, Fore, Style
import rasterio

def dpird_correct(site_name,site_data):
    site_data['time'] = pd.to_datetime(site_data['time'])
    if site_name == "Pemberton":
        site_data["wind_3m_degN"] = (site_data["wind_3m_degN"] + 90) % 360
    if site_name == 'DFES-I Portable':
        site_data['wind_3m_speed'] = np.nan
        site_data['wind_3m_degN'] = np.nan
    if site_name =='DBCA-B Portable':
        site_data['wind_3m_speed'] = np.nan
        site_data['wind_3m_degN'] = np.nan
    if site_name == 'Scaddan':
        invalid_since = pd.to_datetime('2024-2-6 06:45:00')
        site_data.loc[site_data['time']>= invalid_since,'wind_3m_speed'] = np.nan
    # if site_name == 'Glen Eagle':
    #     site_data["wind_speed"] = np.nan
    return site_data

def form_dpird_dataset(
    dataset_path,
    station_coord_path,
    grid_deg,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start,
    end,
):
    features = [
        "airTemperature",
        "apparentAirTemperature",
        "relativeHumidity",
        "dewPoint",
        "panEvaporation",
        "evapotranspiration_shortCrop",
        "evapotranspiration_tallCrop",
        "richardsonUnits",
        "solarExposure",
        "rainfall",
        "deltaT",
        "wetBulb",
        "frostCondition",
        "heatCondition",
        "wind_3m_u",
        "wind_3m_v",
    ]

    rectgrid_lat = np.linspace(
        lat_min, lat_max, num=32, endpoint=False, dtype="float64"
    )
    rectgrid_lon = np.linspace(
        lon_min, lon_max, num=32, endpoint=False, dtype="float64"
    )

    # create an empty uniformed dataset
    time_range = pd.date_range(start=start, end=end, freq="15T")

    dataset = xr.Dataset()
    nan_array = np.full((len(time_range), len(rectgrid_lat), len(rectgrid_lon)), np.nan)
    feature_data = {feature: nan_array.copy() for feature in features}

    # fill in the dpird features
    station_coords = pd.read_csv(station_coord_path)
    for ix, row_i in station_coords.iterrows():
        print("is processing", row_i["name"])
        name, latitude, longitude = row_i["name"], row_i["lat"], row_i["lon"]
        if (
            latitude < lat_min
            or latitude > lat_max
            or longitude < lon_min
            or longitude > lon_max
        ):
            continue
        nearest_latitude_ix = np.argmin(np.abs(rectgrid_lat - latitude))
        nearest_longitude_ix = np.argmin(np.abs(rectgrid_lon - longitude))
        site_data_path = os.path.join(dataset_path, name + ".csv")
        if os.path.exists(site_data_path):
            site_data = pd.read_csv(site_data_path)
            site_data = dpird_correct(row_i['name'],site_data)
        else:
            continue
        site_data["time"] = pd.to_datetime(site_data["time"])
        site_data = site_data[site_data["time"].isin(time_range)]
        site_data["wind_3m_u"], site_data["wind_3m_v"] = calculate_uv_components(
            site_data["wind_3m_speed"].values, site_data["wind_3m_degN"].values
        )

        for jx, row_j in site_data.iterrows():
            time = pd.to_datetime(row_j["time"])
            if row_j[features].isna().all():
                continue
            time_ix = time_range.get_loc(time)
            for feature in features:
                if pd.isna(row_j[feature]):
                    feature_data[feature][
                        time_ix, nearest_latitude_ix, nearest_longitude_ix
                    ] = np.nan
                else:
                    feature_data[feature][
                        time_ix, nearest_latitude_ix, nearest_longitude_ix
                    ] = row_j[feature]

    for feature in features:
        data_array = xr.DataArray(
            data=feature_data[feature],
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_range,
                "latitude": rectgrid_lat,
                "longitude": rectgrid_lon,
            },
            name=feature,
        )
        dataset[feature] = data_array
    return dataset


def raise_value_error(message):
    raise ValueError(f"{Fore.RED}{Style.BRIGHT}Error: {message}{Style.RESET_ALL}")


def create_label_grid(
    dataset_path,
    star_coord_path,
    grid_deg,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start,
    end,
):
    init()
    features = ["wind_10m_u", "wind_10m_v"]

    rectgrid_lat = np.linspace(
        lat_min, lat_max, num=32, endpoint=False, dtype="float64"
    )
    rectgrid_lon = np.linspace(
        lon_min, lon_max, num=32, endpoint=False, dtype="float64"
    )

    # create an empty uniformed dataset
    time_range = pd.date_range(start=start, end=end, freq="15T")

    dataset = xr.Dataset()
    nan_array = np.full((len(time_range), len(rectgrid_lat), len(rectgrid_lon)), np.nan)
    feature_data = {feature: nan_array.copy() for feature in features}

    # fill in the dpird features
    station_coords = pd.read_csv(star_coord_path)
    n = 0
    for ix, row_i in station_coords.iterrows():
        print("is processing", row_i["name"])
        name, latitude, longitude = row_i["name"], row_i["lat"], row_i["lon"]
        if (
            latitude < lat_min
            or latitude > lat_max
            or longitude < lon_min
            or longitude > lon_max
        ):
            continue

        nearest_latitude_ix = np.argmin(np.abs(rectgrid_lat - latitude))
        nearest_longitude_ix = np.argmin(np.abs(rectgrid_lon - longitude))
        site_data_path = os.path.join(dataset_path, name + ".csv")
        if os.path.exists(site_data_path):
            site_data = pd.read_csv(site_data_path)
            site_data = dpird_correct(row_i['name'],site_data)
        else:
            continue
        site_data["time"] = pd.to_datetime(site_data["time"])
        site_data = site_data[site_data["time"].isin(time_range)]
        site_data["wind_10m_u"], site_data["wind_10m_v"] = calculate_uv_components(
            site_data["wind_10m_speed"].values, site_data["wind_10m_degN"].values
        )

        for jx, row_j in site_data.iterrows():
            time = pd.to_datetime(row_j["time"])
            time_ix = time_range.get_loc(time)

            for feature in features:
                if pd.isna(row_j[feature]):
                    feature_data[feature][
                        time_ix, nearest_latitude_ix, nearest_longitude_ix
                    ] = np.nan
                else:
                    feature_data[feature][
                        time_ix, nearest_latitude_ix, nearest_longitude_ix
                    ] = row_j[feature]
        n += 1
    if n == 0:
        raise_value_error(
            f"The grid {lat_min},{lat_max},{lon_min},{lon_max} does not include any ground truth. The process is terminated!!!"
        )
    for feature in features:
        data_array = xr.DataArray(
            data=feature_data[feature],
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_range,
                "latitude": rectgrid_lat,
                "longitude": rectgrid_lon,
            },
            name=feature,
        )
        dataset[feature] = data_array
    return dataset


def create_label3m_grid(
    dataset_path,
    stations_coords_path,
    grid_deg,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start,
    end,
):
    init()
    features = ["wind_3m_u", "wind_3m_v"]

    rectgrid_lat = np.linspace(
        lat_min, lat_max, num=32, endpoint=False, dtype="float64"
    )
    rectgrid_lon = np.linspace(
        lon_min, lon_max, num=32, endpoint=False, dtype="float64"
    )

    # create an empty uniformed dataset
    time_range = pd.date_range(start=start, end=end, freq="15T")

    dataset = xr.Dataset()
    nan_array = np.full((len(time_range), len(rectgrid_lat), len(rectgrid_lon)), np.nan)
    feature_data = {feature: nan_array.copy() for feature in features}

    # fill in the dpird features
    station_coords = pd.read_csv(stations_coords_path)
    n = 0
    for ix, row_i in station_coords.iterrows():

        name, latitude, longitude = row_i["name"], row_i["lat"], row_i["lon"]
        if (
            latitude < lat_min
            or latitude > lat_max
            or longitude < lon_min
            or longitude > lon_max
        ):
            continue
        print("is processing", row_i["name"])
        nearest_latitude_ix = np.argmin(np.abs(rectgrid_lat - latitude))
        nearest_longitude_ix = np.argmin(np.abs(rectgrid_lon - longitude))
        site_data_path = os.path.join(dataset_path, name + ".csv")
        if os.path.exists(site_data_path):
            site_data = pd.read_csv(site_data_path)
            site_data = dpird_correct(row_i['name'],site_data)
        else:
            continue
        site_data["time"] = pd.to_datetime(site_data["time"])
        site_data = site_data[site_data["time"].isin(time_range)]
        site_data["wind_3m_u"], site_data["wind_3m_v"] = calculate_uv_components(
            site_data["wind_3m_speed"].values, site_data["wind_3m_degN"].values
        )

        for jx, row_j in site_data.iterrows():
            time = pd.to_datetime(row_j["time"])
            time_ix = time_range.get_loc(time)

            for feature in features:
                if pd.isna(row_j[feature]):
                    feature_data[feature][
                        time_ix, nearest_latitude_ix, nearest_longitude_ix
                    ] = np.nan
                else:
                    feature_data[feature][
                        time_ix, nearest_latitude_ix, nearest_longitude_ix
                    ] = row_j[feature]
        n += 1
    if n == 0:
        raise_value_error(
            f"The grid {lat_min},{lat_max},{lon_min},{lon_max} does not include any ground truth. The process is terminated!!!"
        )
    for feature in features:
        data_array = xr.DataArray(
            data=feature_data[feature],
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_range,
                "latitude": rectgrid_lat,
                "longitude": rectgrid_lon,
            },
            name=feature,
        )
        dataset[feature] = data_array
    return dataset


def get_alt(terrain_data, lat, lon):

    lat_max = -9.00  # - 0.0025/2
    lat_min = -43.7425  # + 0.0025/2
    lon_min = 112.90  # + 0.0025/2
    lon_max = 154.00  # - 0.0025/2
    rectgrid_lat = np.linspace(
        lat_max, lat_min, num=terrain_data.shape[0], endpoint=True, dtype="float64"
    )
    rectgrid_lon = np.linspace(
        lon_min, lon_max, num=terrain_data.shape[1], endpoint=True, dtype="float64"
    )
    nearest_latitude_ix = np.argmin(np.abs(rectgrid_lat - lat))
    nearest_longitude_ix = np.argmin(np.abs(rectgrid_lon - lon))

    alt = terrain_data[nearest_latitude_ix, nearest_longitude_ix]
    return alt


def create_terrain_grid(
    dataset_path,
    grid_deg,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start,
    end,
):
    with rasterio.open(dataset_path) as src:
        terrain_data = src.read(1)  # Read the first band

    rectgrid_lat = np.linspace(
        lat_min, lat_max, num=32, endpoint=False, dtype="float64"
    )
    rectgrid_lon = np.linspace(
        lon_min, lon_max, num=32, endpoint=False, dtype="float64"
    )

    # create an empty uniformed dataset
    time_range = pd.date_range(start=start, end=end, freq="15T")

    dataset = xr.Dataset()
    nan_array = np.full((len(time_range), len(rectgrid_lat), len(rectgrid_lon)), np.nan)
    for ilat, lat in enumerate(rectgrid_lat):
        for ilon, lon in enumerate(rectgrid_lon):
            alt = get_alt(terrain_data, lat, lon)
            if alt < 0:
                alt = 0
            nan_array[:, ilat, ilon] = alt

    data_array = xr.DataArray(
        data=nan_array,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time_range,
            "latitude": rectgrid_lat,
            "longitude": rectgrid_lon,
        },
        name="terrain",
    )
    dataset["terrain"] = data_array
    return dataset


def create_label_ix(star_coord_path, ds: xr.Dataset, args):
    lats_ix, lons_ix = [], []
    station_coords = pd.read_csv(star_coord_path)
    star_coords = station_coords.copy()
    for ix, row_i in station_coords.iterrows():
        if (args.lat_min <= row_i["lat"] <= args.lat_max) and (
            args.lon_min <= row_i["lon"] <= args.lon_max
        ):
            nearest_latitude = ds["latitude"].sel(
                latitude=row_i["lat"], method="nearest"
            )
            nearest_longitude = ds["longitude"].sel(
                longitude=row_i["lon"], method="nearest"
            )
            lat_ix = np.where(ds["latitude"].values == nearest_latitude.values)[0][0]
            lon_ix = np.where(ds["longitude"].values == nearest_longitude.values)[0][0]
            lats_ix.append(lat_ix)
            lons_ix.append(lon_ix)
        else:
            star_coords = star_coords.drop(ix)

    star_coords["lat_ix"] = lats_ix
    star_coords["lon_ix"] = lons_ix
    return star_coords

def create_label3m_ix(station3m_coord_path, ds: xr.Dataset, args):
    lats_ix, lons_ix = [], []
    station_coords = pd.read_csv(station3m_coord_path)
    station3m_coords = station_coords.copy()
    for ix, row_i in station_coords.iterrows():
        if (args.lat_min <= row_i["lat"] <= args.lat_max) and (
                args.lon_min <= row_i["lon"] <= args.lon_max
        ):
            nearest_latitude = ds["latitude"].sel(
                latitude=row_i["lat"], method="nearest"
            )
            nearest_longitude = ds["longitude"].sel(
                longitude=row_i["lon"], method="nearest"
            )
            lat_ix = np.where(ds["latitude"].values == nearest_latitude.values)[0][0]
            lon_ix = np.where(ds["longitude"].values == nearest_longitude.values)[0][0]
            lats_ix.append(lat_ix)
            lons_ix.append(lon_ix)
        else:
            station3m_coords = station3m_coords.drop(ix)

    station3m_coords["lat_ix"] = lats_ix
    station3m_coords["lon_ix"] = lons_ix
    return station3m_coords


def uniform_cds(cds: xr.Dataset, start, end):
    # cds should be an original xarray dataset downloaded from era
    cds = cds.sortby("time")
    cds = cds.sel(time=slice(start, end))
    # retrieve_time = pd.Timestamp(cds["time"].max().values.item()) + datetime.timedelta(
    #     days=5
    # )
    # three_months_ago = retrieve_time - datetime.timedelta(days=90)
    # expver_5_data = cds.sel(expver=5, time=slice(three_months_ago, None))
    # expver_1_data = cds.sel(expver=1, time=slice(None, three_months_ago))
    retrieve_time = pd.Timestamp(cds["time"].max().values.item()) - datetime.timedelta(
        days=31
    )
    expver_5_data = cds.sel(expver=5, time=slice(retrieve_time, None))
    expver_1_data = cds.sel(expver=1, time=slice(None, retrieve_time))

    unified_data = xr.concat([expver_1_data, expver_5_data], dim="time")
    unified_data = unified_data.dropna(dim="time", how="all")
    unified_data = unified_data.drop_vars("expver")

    variables = list(unified_data.data_vars.keys())
    # the cds provides wind speed in m/s
    if "u10" in variables:
        unified_data["u10"] = mps_to_kmph(-unified_data["u10"])
    if "v10" in variables:
        unified_data["v10"] = mps_to_kmph(-unified_data["v10"])
    if "msl" in variables:
        unified_data["msl"] = unified_data["msl"] / 100
    return unified_data


def mps_to_kmph(speed_mps):
    # Conversion factor from m/s to km/h
    conversion_factor = 3.6
    speed_kmph = speed_mps * conversion_factor  # CHECK here
    return speed_kmph


def calculate_uv_components(speed, direction):
    # direction is in degrees
    direction_radians = np.radians(direction)
    u_component = speed * np.cos(direction_radians)  # from east
    v_component = speed * np.sin(direction_radians)  # from north
    return u_component, v_component


def uniform_met(met: pd.DataFrame):
    # met should be an original dataframe
    met["update_time"] = pd.to_datetime(met["update_time"])
    met["time"] = pd.to_datetime(met["time"])
    met_sorted = met.sort_values(by=["time", "update_time"], ascending=[True, False])
    met_unique_latest = met_sorted.drop_duplicates(subset=["lat", "lon", "time"])
    met_unique_latest.loc[:, "wind_speed"] = met_unique_latest["wind_speed"].apply(
        mps_to_kmph
    )
    met_unique_latest[["met_u10", "met_v10"]] = calculate_uv_components(
        met_unique_latest["wind_speed"], met_unique_latest["wind_from_direction"]
    )
    met_unique_latest = met_unique_latest.drop(
        ["wind_speed", "wind_from_direction"], axis=1
    )
    met_unique_latest.rename(
        columns={
            "lat": "latitude",
            "lon": "longitude",
            "air_pressure_at_sea_level": "met_msl",
            "air_temperature": "met_air_temperature",
            "cloud_area_fraction": "met_could_area_fraction",
            "relative_humidity": "met_relative_humidity",
        },
        inplace=True,
    )
    met_unique_latest.set_index(["time", "latitude", "longitude"], inplace=True)
    uniformed_met = met_unique_latest.to_xarray()
    return uniformed_met


def interp_time(
    data: xr.Dataset,
    interval: str,
    variables: List[str] = None,
    start: str = None,
    end: str = None,
):
    if variables:
        data = data[variables]
    else:
        variables = list(data.data_vars.keys())
    if start or end:
        data = data.sel(time=slice(start, end))
    data_interp = data.resample(time=interval).interpolate("linear")
    return data_interp


def interp_space(
    data: xr.Dataset,
    interp_mode,
    grid_deg,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
):
    # this interpolation is only for ECMWF or MET
    # DPIRD observations should not be interpolated spatially
    assert interp_mode in ["krig", "linear"]
    interp = Interpolation_model_grid(
        data,
        grid_deg=grid_deg,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    if interp_mode == "krig":
        interped_dataset = interp.krig_3D_vars_xr()
    else:
        interped_dataset = interp.ln_interp_3D_vars_xr()
    return interped_dataset
