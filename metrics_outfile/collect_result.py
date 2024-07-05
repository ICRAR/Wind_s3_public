import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import xarray as xr
import os
import glob


def convert_uv_to_vdir(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction_radians = np.arctan2(v, u)
    direction_degrees = np.degrees(direction_radians)

    # Convert negative angles to positive angles
    direction_degrees = (direction_degrees + 360) % 360
    # Calculate sine and cosine
    sin_value = np.sin(direction_radians)
    cos_value = np.cos(direction_radians)
    return speed, direction_degrees, sin_value, cos_value


def get_site_horizon_result(args,file_path):
    print(args.result_save_path)

    df = pd.read_csv(args.result_save_path)
    T_hr = int(args.T_hr)
    L_hr = int(args.L_hr)

    star = pd.read_csv(args.star_coord_ix)
    df["time"] = pd.to_datetime(df["time"])
    df["time_delta"] = pd.to_datetime(df["time_delta"])

    # ds_ecmwf = xr.open_dataset(args.cds_dest_path)
    # if 'ecmwf_u' not in df.columns:
    #     df['ecmwf_u'] = np.nan
    #     df['ecmwf_u'] = np.nan
    #     for i,row in df.iterrows():
    #         latitude = star[star['name']==row['site']]['lat'].values
    #         longitude = star[star['name']==row['site']]['lon'].values
    #         df.loc[i,'ecmwf_u'] = ds_ecmwf['u10'].sel(time=row['time_delta'],latitude=latitude,\
    #                                              longitude=longitude,method='nearest').values
    #
    #         df.loc[i,'ecmwf_v'] = ds_ecmwf['v10'].sel(time=row['time_delta'],latitude=latitude,\
    #                           longitude=longitude,method='nearest').values
    if "truth_speed" not in df.columns:
        (
            df["truth_speed"],
            df["truth_deg"],
            df["truth_sin"],
            df["truth_cos"],
        ) = convert_uv_to_vdir(df["label_u"].values, df["label_v"].values)
    if "pred_speed" not in df.columns:
        (
            df["pred_speed"],
            df["pred_deg"],
            df["pred_sin"],
            df["pred_cos"],
        ) = convert_uv_to_vdir(df["pred_u"].values, df["pred_v"].values)
    if "ecmwf_speed" not in df.columns:
        (
            df["ecmwf_speed"],
            df["ecmwf_deg"],
            df["ecmwf_sin"],
            df["ecmwf_cos"],
        ) = convert_uv_to_vdir(df["ecmwf_u"].values, df["ecmwf_v"].values)

    df.to_csv(args.result_save_path, index=False)

    sites = star["name"]
    results = pd.DataFrame()
    for site in sites:
        result = pd.DataFrame()
        (
            hrlist,
            mse_speed_list,
            mse_sin_list,
            mse_cos_list,
            mse_ecmwf_speed_list,
            mse_ecmwf_sin_list,
            mse_ecmwf_cos_list,
        ) = ([], [], [], [], [], [], [])
        df_site = df[df["site"] == site]
        horizon_zero = (
            df_site["time"]
            + pd.to_timedelta(T_hr, unit="h")
            - pd.to_timedelta(L_hr, unit="h")
        )
        df_site = df_site[df_site["time_delta"] >= horizon_zero]
        df_site["horizon"] = df_site["time_delta"] - horizon_zero

        horizons = df_site["horizon"].unique()
        for horizon in horizons:
            hrlist.append(horizon)
            df_site_hr = df_site[df_site["horizon"] == horizon]
            df_site_hr = df_site_hr.dropna()
            mse_speed_hr = mse(df_site_hr["truth_speed"], df_site_hr["pred_speed"])
            mse_sin_hr = mse(df_site_hr["truth_sin"], df_site_hr["pred_sin"])
            mse_cos_hr = mse(df_site_hr["truth_cos"], df_site_hr["pred_cos"])
            mse_ecmwf_speed_hr = mse(
                df_site_hr["truth_speed"], df_site_hr["ecmwf_speed"]
            )
            mse_ecmwf_sin_hr = mse(df_site_hr["truth_sin"], df_site_hr["ecmwf_sin"])
            mse_ecmwf_cos_hr = mse(df_site_hr["truth_cos"], df_site_hr["ecmwf_cos"])

            mse_speed_list.append(mse_speed_hr)
            mse_sin_list.append(mse_sin_hr)
            mse_cos_list.append(mse_cos_hr)
            mse_ecmwf_speed_list.append(mse_ecmwf_speed_hr)
            mse_ecmwf_sin_list.append(mse_ecmwf_sin_hr)
            mse_ecmwf_cos_list.append(mse_ecmwf_cos_hr)
        result["horizon"] = hrlist
        result["mse_speed"] = mse_speed_list
        result["mse_sin"] = mse_sin_list
        result["mse_cos"] = mse_cos_list
        result["mse_ecmwf_speed"] = mse_ecmwf_speed_list
        result["mse_ecmwf_sin"] = mse_ecmwf_sin_list
        result["mse_ecmwf_cos"] = mse_ecmwf_cos_list
        result["site"] = site
        results = pd.concat([results, result])

    results.to_csv(file_path, index=False)


def generate_result_csv(args):
    file_path = args.result_save_path.split(".csv")[0] + "_result.csv"
    if not os.path.exists(file_path):
        get_site_horizon_result(args, file_path=file_path)

def generate_correlation_csv(args,pred_path):

    df = pd.read_csv(pred_path)
    T_hr = int(args.T_hr)
    L_hr = int(args.L_hr)

    stations3m = pd.read_csv(args.station3m_coord_ix)
    df["time"] = pd.to_datetime(df["time"],errors='coerce')
    df["time_delta"] = pd.to_datetime(df["time_delta"],errors='coerce')

    sites = stations3m["name"]
    results = pd.DataFrame()
    for site in sites:
        print('collect result for',site)
        result = pd.DataFrame()
        (
            hrlist,
            r_u_list,
            r_v_list,
            ecmwf_r_u_list,
            ecmwf_r_v_list
        ) = ([], [], [], [], [])
        df_site = df[df["site"] == site]
        horizon_zero = (
                df_site["time"]
                + pd.to_timedelta(T_hr, unit="h")
                - pd.to_timedelta(L_hr, unit="h")
        )
        df_site = df_site[df_site["time_delta"] >= horizon_zero]
        df_site["horizon"] = df_site["time_delta"] - horizon_zero

        horizons = df_site["horizon"].unique()
        for horizon in horizons:
            hrlist.append(horizon)
            df_site_hr = df_site[df_site["horizon"] == horizon]
            df_site_hr = df_site_hr.dropna()
            r_u = df_site_hr['label3m_u'].corr(df_site_hr['pred_u'])
            r_v = df_site_hr['label3m_v'].corr(df_site_hr['pred_v'])
            ecmwf_u = df_site_hr['label3m_u'].corr(df_site_hr['ecmwf_u'])
            ecmwf_v = df_site_hr['label3m_v'].corr(df_site_hr['ecmwf_v'])

            r_u_list.append(r_u)
            r_v_list.append(r_v)
            ecmwf_r_u_list.append(ecmwf_u)
            ecmwf_r_v_list.append(ecmwf_v)

        result["horizon"] = hrlist
        result["r_u"] = r_u_list
        result["r_v"] = r_v_list
        result["ecmwf_r_u"] = ecmwf_r_u_list
        result["ecmwf_r_v"] = ecmwf_r_v_list
        result["site"] = site
        results = pd.concat([results, result])
    file_path = pred_path.split(".csv")[0] + "_result.csv"
    results.to_csv(file_path, index=False)


def generate_result_csv_all(args):
    root_path = args.result_path.split("result_ST")[0] + "result_ST/"
    # Use glob to find files with names ending in "pred.csv" recursively
    file_pattern = os.path.join(root_path, "**", "*pred.csv")
    matching_files = glob.glob(file_pattern, recursive=True)

    # Display the matching files
    for file_path in matching_files:
        if not os.path.exists(file_path.split(".csv")[0] + "_result.csv"):
            get_site_horizon_result(args, file_path)
