import numpy as np
import pandas as pd
import xarray as xr
import torch
import mlflow
import json
import datetime
from sklearn.preprocessing import StandardScaler


def out_per_df(
    args, out: np.ndarray, y: np.ndarray, time: np.ndarray, x_ecmwf: np.ndarray
) -> pd.DataFrame:
    # print(y.shape, out.shape, time.shape, x_ecmwf.shape)
    df_list = []
    time_range = pd.to_datetime(time, unit="ns", utc=True)

    star_loc = pd.read_csv(args.star_coord_ix)
    for i, star in star_loc.iterrows():
        label_u = y[0, 0, :, star["lat_ix"], star["lon_ix"]]
        label_v = y[0, 1, :, star["lat_ix"], star["lon_ix"]]
        pred_u = out[0, 0, :, star["lat_ix"], star["lon_ix"]]
        pred_v = out[0, 1, :, star["lat_ix"], star["lon_ix"]]
        ecmwf_u = x_ecmwf[0, 0, :, star["lat_ix"], star["lon_ix"]]
        ecmwf_v = x_ecmwf[0, 1, :, star["lat_ix"], star["lon_ix"]]
        length = len(label_u)
        df_i = pd.DataFrame(
            {
                "time_delta": time_range.values[0],
                "time": [time_range.values[0, 0]] * length,
                "site": [star["name"]] * length,
                "label_u": label_u.tolist(),
                "label_v": label_v.tolist(),
                "pred_u": pred_u.tolist(),
                "pred_v": pred_v.tolist(),
                "ecmwf_u": ecmwf_u.tolist(),
                "ecmwf_v": ecmwf_v.tolist(),
            }
        )
        df_list.append(df_i)
    df = pd.concat(df_list, ignore_index=True)
    return df


def out_per3m_df(
    args,
    out: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    x_ecmwf: np.ndarray,
    y3m: np.ndarray,
) -> pd.DataFrame:
    # print(y.shape, out.shape, time.shape, x_ecmwf.shape,y3m.shape)
    df_list = []
    time_range = pd.to_datetime(time, unit="ns", utc=True)

    station3m_loc = pd.read_csv(args.station3m_coord_ix)
    for i, station3m in station3m_loc.iterrows():
        label_u = y3m[0, 0, :, station3m["lat_ix"], station3m["lon_ix"]]
        label_v = y3m[0, 1, :, station3m["lat_ix"], station3m["lon_ix"]]
        pred_u = out[0, 0, :, station3m["lat_ix"], station3m["lon_ix"]]
        pred_v = out[0, 1, :, station3m["lat_ix"], station3m["lon_ix"]]
        ecmwf_u = x_ecmwf[0, 0, :, station3m["lat_ix"], station3m["lon_ix"]]
        ecmwf_v = x_ecmwf[0, 1, :, station3m["lat_ix"], station3m["lon_ix"]]
        length = len(label_u)
        df_i = pd.DataFrame(
            {
                "time_delta": time_range.values[0],
                "time": [time_range.values[0, 0]] * length,
                "site": [station3m["name"]] * length,
                "label3m_u": label_u.tolist(),
                "label3m_v": label_v.tolist(),
                "pred_u": pred_u.tolist(),
                "pred_v": pred_v.tolist(),
                "ecmwf_u": ecmwf_u.tolist(),
                "ecmwf_v": ecmwf_v.tolist(),
            }
        )
        df_list.append(df_i)
    df = pd.concat(df_list, ignore_index=True)
    return df


def out_per_df_S12(y_hat, y, y_time, x_end, scaler) -> pd.DataFrame:
    df = pd.DataFrame()
    df["time"] = pd.to_datetime(x_end[0], unit="s", utc=True)
    df["time_delta"] = pd.to_datetime(y_time[0], unit="ns", utc=True)

    df[["true_speed", "true_degN"]] = scaler.inverse_transform(y[0, :, :])
    df[["pred_speed", "pred_degN"]] = scaler.inverse_transform(y_hat[0, :, :])
    # only need the future data
    df = df[df["time_delta"] >= df["time"]].reset_index(drop=True)

    return df


def out_per_df_S12_h1(y_hat, y, y_time, x_end, scaler) -> pd.DataFrame:
    df = pd.DataFrame()
    df["time"] = pd.to_datetime(x_end[0], unit="s", utc=True)
    df["time_delta"] = pd.to_datetime(y_time[0], unit="ns", utc=True)

    df[["true_speed", "true_degN"]] = scaler.inverse_transform(y[0, :, :])
    df[["pred_speed", "pred_degN"]] = scaler.inverse_transform(y_hat[:, :])
    # only need the future data
    df = df[df["time_delta"] >= df["time"]].reset_index(drop=True)

    return df


def out_per_df_S12_h1m1(y_hat, y, y_time, L, y_mean, y_std) -> pd.DataFrame:
    df = pd.DataFrame()
    df["time_delta"] = [datetime.datetime.fromtimestamp(int(y_time))]
    df["time"] = [
        datetime.datetime.fromtimestamp(int(y_time)) - datetime.timedelta(minutes=L)
    ]
    scaler = StandardScaler()
    scaler.mean_ = y_mean
    scaler.scale_ = y_std
    df[["true_speed", "true_degN"]] = scaler.inverse_transform(y)
    df[["pred_speed", "pred_degN"]] = scaler.inverse_transform(y_hat)

    return df


def out_ds(
    args, out: np.ndarray, y: np.ndarray, time: np.ndarray, x_ecmwf: np.ndarray
) -> xr.Dataset:
    # get the horizon 0 onwards
    n_timeframe = (args.T_hr - args.L_hr) * 4
    time_to = pd.to_datetime(time[0], unit="ns", utc=True)[n_timeframe - 1 :]
    length = len(time_to)
    time_to = pd.to_datetime(time_to, unit="ns", utc=True)

    rectgrid_lat = np.linspace(
        args.lat_min, args.lat_max, num=32, endpoint=False, dtype="float64"
    )
    rectgrid_lon = np.linspace(
        args.lon_min, args.lon_max, num=32, endpoint=False, dtype="float64"
    )
    time_since = time_to[0]
    print(time_to[0])
    # label_u = np.repeat(y[0, 0, n_timeframe-1:][np.newaxis, ...], length, axis=0)
    # label_v = np.repeat(y[0, 1, n_timeframe-1:][np.newaxis, ...], length, axis=0)
    # pred_u = np.repeat(out[0, 0, n_timeframe-1:][np.newaxis, ...], length, axis=0)
    # pred_v = np.repeat(out[0, 1, n_timeframe-1:][np.newaxis, ...], length, axis=0)
    # ecmwf_u = np.repeat(x_ecmwf[0, 0, n_timeframe-1:][np.newaxis, ...], length, axis=0)
    # ecmwf_v = np.repeat(x_ecmwf[0, 1, n_timeframe-1:][np.newaxis, ...], length, axis=0)

    label_u = y[0, 0, n_timeframe - 1 :]
    label_v = y[0, 1, n_timeframe - 1 :]
    pred_u = out[0, 0, n_timeframe - 1 :]
    pred_v = out[0, 1, n_timeframe - 1 :]
    ecmwf_u = x_ecmwf[0, 0, n_timeframe - 1 :]
    ecmwf_v = x_ecmwf[0, 1, n_timeframe - 1 :]
    ds = xr.Dataset(
        {
            "label_u": (["time_to", "latitude", "longitude"], label_u),
            "label_v": (["time_to", "latitude", "longitude"], label_v),
            "pred_u": (["time_to", "latitude", "longitude"], pred_u),
            "pred_v": (["time_to", "latitude", "longitude"], pred_v),
            "ecmwf_u": (["time_to", "latitude", "longitude"], ecmwf_u),
            "ecmwf_v": (["time_to", "latitude", "longitude"], ecmwf_v),
        },
        coords={
            "time_since": time_since,
            "time_to": time_to,
            "latitude": rectgrid_lat,
            "longitude": rectgrid_lon,
        },
    )
    return ds


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)
