import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pytz, os, math, rasterio
from datetime import datetime
import matplotlib.dates as mdates
from numpy.ma import masked_array
import matplotlib.cm as cm
import argparse
from metrics_outfile.collect_result import generate_correlation_csv

ALL_STATION_COORDS = pd.read_csv("../../../../data_prep_ST/make_grid/all_station_coordinates.csv")
STAR_STATION_CORRDS = pd.read_csv("../../../../data_prep_ST/make_grid/stations_available_label_coordinates.csv")
# dem_data_path = "../../../../data_prep_ST/terrain/DEM-9S/Data_9secDEM_D8/dem-9s/dem-9s.tif"
dem_data_path = "/mnt/science1/fchen/dataset_terrain/dem-9s.tif"

def files(key):
    # starfile = f"../35.4_32.0_115.0_118.4_7feat_terrain_2022_2023_{key}_predstar.csv"
    # station3mfile = f"../35.4_32.0_115.0_118.4_7feat_terrain_2022_2023_{key}_predstation3m_result.csv"
    starfile = f"/mnt/science1/fchen/result_ST/lat32_lon115_3d4_2022_2023/35.4_32.0_115.0_118.4_7feat_terrain_2022_2023_{key}_predstar.csv"
    def aggregate_predstation3m_updatecor():

        predstation3m_pred_path = f'/mnt/science1/fchen/result_ST/lat32_lon115_3d4_2022_2023/35.4_32.0_115.0_118.4_7feat_terrain_2022_2023_{key}_predstation3m.csv'
        station3mfile = predstation3m_pred_path.split('.csv')[0]+'_result.csv'
        if not os.path.exists(station3mfile):
            if not os.path.exists(predstation3m_pred_path):
                input_dir = os.path.dirname(predstation3m_pred_path)
                predstation3m_paths = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if f.endswith("predstation3m.csv")]

                dfs = pd.DataFrame()

                for path in predstation3m_paths:
                    df = pd.read_csv(path)
                    dfs = pd.concat([dfs, df])
                dfs.to_csv(predstation3m_pred_path,index=False)
            parser = argparse.ArgumentParser(description="Description of your script")
            parser.add_argument('--T_hr', type=int, default=48)
            parser.add_argument('--L_hr', type=int, default=4)
            parser.add_argument('--station3m_coord_ix', type=str,
                                default='/mnt/science1/fchen/result_ST/lat32_lon115_3d4_2022_2023/station3m_loc_35.4_32.0_115.0_118.4_7feat_terrain_2022_2023_loss2.csv')
            args = parser.parse_args()
            generate_correlation_csv(args,pred_path=predstation3m_pred_path)
        return station3mfile
    station3mfile = aggregate_predstation3m_updatecor()
    return starfile,station3mfile

def preprocess(file):
    df = pd.read_csv(file)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["time_delta"] = pd.to_datetime(df["time_delta"], utc=True)
    horizon_zero = (
        df["time"] + pd.to_timedelta(48, unit="h") - pd.to_timedelta(4, unit="h")
    )
    df = df[df["time_delta"] >= horizon_zero]
    df["horizon"] = df["time_delta"] - horizon_zero
    df["horizon"] = pd.to_timedelta(df["horizon"]).dt.total_seconds() / 3600
    df = df[df["horizon"] >= 0]
    # target_timezone = pytz.timezone("Australia/Perth")
    # df["time"] = df["time"].dt.tz_convert(target_timezone)
    # df["time_delta"] = df["time_delta"].dt.tz_convert(target_timezone)

    rename_site = {
        "Dumbleyung GRDC": "DU002",
        "Jarrahdale 2": "JA002",
        "Katanning GRDC": "KA002",
        "Gnowangerup GRDC": "GN002",
        "Mount Barker": "MB",
        "Quinninup": "QP001",
        "Pemberton": "PM",
    }
    df["site"] = df["site"].replace(rename_site)
    df["ecmwf_u"] = df["ecmwf_u"]
    df["ecmwf_v"] = df[
        "ecmwf_v"
    ]  # wind speed in km/h, here is to fix the bug in preprocess
    return df


def clip_data_type(df, type):
    if type == "all":
        df = df
    elif type == "winterday":
        start_time = datetime.strptime("07:15", "%H:%M").time()
        end_time = datetime.strptime("17:30", "%H:%M").time()
        df = df[
            (df["time_delta"].dt.month >= 6)
            & (df["time_delta"].dt.month <= 9)
            & (df["time_delta"].dt.time >= start_time)
            & (df["time_delta"].dt.time <= end_time)
        ]
    elif type == "winternight":
        start_time = datetime.strptime("07:15", "%H:%M").time()
        end_time = datetime.strptime("17:30", "%H:%M").time()
        df = df[
            (df["time_delta"].dt.month >= 6)
            & (df["time_delta"].dt.month <= 9)
            & (
                (df["time_delta"].dt.time < start_time)
                | (df["time_delta"].dt.time > end_time)
            )
        ]
    elif type == "summerday":
        start_time = datetime.strptime("05:00", "%H:%M").time()
        end_time = datetime.strptime("19:30", "%H:%M").time()
        df = df[
            ((df["time_delta"].dt.month >= 11) | (df["time_delta"].dt.month <= 2))
            & (df["time_delta"].dt.time >= start_time)
            & (df["time_delta"].dt.time <= end_time)
        ]
    elif type == "summernight":
        start_time = datetime.strptime("05:00", "%H:%M").time()
        end_time = datetime.strptime("19:30", "%H:%M").time()
        df = df[
            ((df["time_delta"].dt.month >= 11) | (df["time_delta"].dt.month <= 2))
            & (
                (df["time_delta"].dt.time < start_time)
                | (df["time_delta"].dt.time > end_time)
            )
        ]
    else:
        raise ValueError(
            "type must be in all, winterday, winternight, summerday or summernight"
        )
    return df


def get_result(df, type):
    df = clip_data_type(df, type)
    rs = pd.DataFrame()
    horizons = df["horizon"].unique()
    sites = df["site"].unique()
    for site in sites:
        for h in horizons:
            for model in ["ABED", "ECMWF"]:
                rs_per = pd.DataFrame()
                data = df[(df["site"] == site) & (df["horizon"] == h)]
                rs_per["site"] = [site]
                rs_per["horizon"] = [h]
                rs_per["model"] = [model]
                if model == "ABED":
                    rs_per["MAE_speed"] = mean_absolute_error(
                        data["truth_speed"], data["pred_speed"]
                    )
                    rs_per["RMSE_speed"] = root_mean_squared_error(
                        data["truth_speed"], data["pred_speed"]
                    )
                    rs_per["MAE_sine"] = mean_absolute_error(
                        data["truth_sin"], data["pred_sin"]
                    )
                    rs_per["RMSE_sine"] = root_mean_squared_error(
                        data["truth_sin"], data["pred_sin"]
                    )
                    rs_per["MAE_cosine"] = mean_absolute_error(
                        data["truth_cos"], data["pred_cos"]
                    )
                    rs_per["RMSE_cosine"] = root_mean_squared_error(
                        data["truth_cos"], data["pred_cos"]
                    )
                    rs_per["MAE_u10"] = mean_absolute_error(
                        data["label_u"], data["pred_u"]
                    )
                    rs_per["RMSE_u10"] = root_mean_squared_error(
                        data["label_u"], data["pred_u"]
                    )
                    rs_per["MAE_v10"] = mean_absolute_error(
                        data["label_v"], data["pred_v"]
                    )
                    rs_per["RMSE_v10"] = root_mean_squared_error(
                        data["label_v"], data["pred_v"]
                    )
                else:
                    rs_per["MAE_speed"] = mean_absolute_error(
                        data["truth_speed"], data["ecmwf_speed"]
                    )
                    rs_per["RMSE_speed"] = root_mean_squared_error(
                        data["truth_speed"], data["ecmwf_speed"]
                    )
                    rs_per["MAE_sine"] = mean_absolute_error(
                        data["truth_sin"], data["ecmwf_sin"]
                    )
                    rs_per["RMSE_sine"] = root_mean_squared_error(
                        data["truth_sin"], data["ecmwf_sin"]
                    )
                    rs_per["MAE_cosine"] = mean_absolute_error(
                        data["truth_cos"], data["ecmwf_cos"]
                    )
                    rs_per["RMSE_cosine"] = root_mean_squared_error(
                        data["truth_cos"], data["ecmwf_cos"]
                    )
                    rs_per["MAE_u10"] = mean_absolute_error(
                        data["label_u"], data["ecmwf_u"]
                    )
                    rs_per["RMSE_u10"] = root_mean_squared_error(
                        data["label_u"], data["ecmwf_u"]
                    )
                    rs_per["MAE_v10"] = mean_absolute_error(
                        data["label_v"], data["ecmwf_v"]
                    )
                    rs_per["RMSE_v10"] = root_mean_squared_error(
                        data["label_v"], data["ecmwf_v"]
                    )
                rs = pd.concat([rs, rs_per])
    return rs


def get_rs(rspath,file,type,key):

    df = preprocess(file)
    rs = get_result(df, type)
    rs.to_csv(rspath, index=False)
    return rs


def plot6in1(rs, plot_path):
    site_colors = {
        "DU002": "blue",
        "GN002": "green",
        "KA002": "red",
        "JA002": "purple",
        "MB": "orange",
        "PM": "cyan",
        "QP001": "pink",
    }
    # Define line styles for each model
    model_line_styles = {"ABED": "-", "ECMWF": "--"}
    sns.set(style="whitegrid")
    # Create a 2x2 plot
    fig, axs = plt.subplots(
        6, 2, figsize=(6, 10), gridspec_kw={"height_ratios": [5, 5,5,5,5, 2]}
    )

    # Plot MAE_u10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_u10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("MAE_u10 (km/h)")
    axs[0, 0].set_ylabel("")
    axs[0, 0].set_xlabel("")
    axs[0, 0].legend().set_visible(False)
    axs[0, 0].set_xlim(0, 8)  # Set x-axis limit
    # Plot MAE_v10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_v10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[1,0],
    )
    axs[1,0].set_title("MAE_v10 (km/h)")
    axs[1,0].set_ylabel("")
    axs[1,0].set_xlabel("")
    axs[1,0].legend().set_visible(False)
    axs[1,0].set_xlim(0, 8)  # Set x-axis limit
    # Plot RMSE_u10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_u10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("RMSE_u10 (km/h)")
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_xlabel("")
    axs[0, 1].legend().set_visible(False)
    axs[0, 1].set_xlim(0, 8)  # Set x-axis limit
    # Plot RMSE_v10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_v10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[1, 1],
    )
    axs[1, 1].set_title("RMSE_v10 (km/h)")
    axs[1, 1].set_ylabel("")
    axs[1, 1].set_xlabel("")
    axs[1, 1].legend().set_visible(False)
    axs[1, 1].set_xlim(0, 8)  # Set x-axis limit
    # Plot MAE_speed
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_speed",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[2, 0],
    )
    axs[2, 0].set_title("MAE_speed (km/h)")
    axs[2, 0].set_ylabel("")
    axs[2, 0].set_xlabel("")
    axs[2, 0].legend().set_visible(False)
    axs[2, 0].set_xlim(0, 8)  # Set x-axis limit

    # Plot RMSE_speed
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_speed",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[2, 1],
    )
    axs[2, 1].set_title("RMSE_speed (km/h)")
    axs[2, 1].set_ylabel("")
    axs[2, 1].set_xlabel("")
    axs[2, 1].legend().set_visible(False)
    axs[2, 1].set_xlim(0, 8)  # Set x-axis limit

    # Plot MAE_sine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_sine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[3, 0],
    )
    axs[3, 0].set_title("MAE_sine")
    axs[3, 0].set_ylabel("")
    axs[3, 0].set_xlabel("")
    axs[3, 0].legend().set_visible(False)
    axs[3, 0].set_xlim(0, 8)  # Set x-axis limit

    # Plot RMSE_sine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_sine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[3, 1],
    )
    axs[3, 1].set_title("RMSE_sine")
    axs[3, 1].set_ylabel("")
    axs[3, 1].set_xlabel("")
    axs[3, 1].legend().set_visible(False)
    axs[3, 1].set_xlim(0, 8)  # Set x-axis limit

    # Plot MAE_cosine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_cosine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[4, 0],
    )
    axs[4, 0].set_title("MAE_cosine")
    axs[4, 0].set_ylabel("")
    axs[4, 0].set_xlabel("horizon (hr)")
    axs[4, 0].legend().set_visible(False)
    axs[4, 0].set_xlim(0, 8)  # Set x-axis limit

    # Plot RMSE_cosine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_cosine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[4, 1],
    )
    axs[4, 1].set_title("RMSE_cosine")
    axs[4, 1].set_ylabel("")
    axs[4, 1].set_xlabel("horizon (hr)")
    axs[4, 1].legend().set_visible(False)
    axs[4, 1].set_xlim(0, 8)  # Set x-axis limit

    # Hide the last row of subplots
    axs[5, 0].axis("off")
    axs[5, 1].axis("off")

    # Adjust y ticks for right subplots
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    # Adjust layout
    plt.subplots_adjust(hspace=0.2)

    # Create legend for colors
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend1 = plt.legend(
        handles[:-3],
        labels[:-3],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(-0.11, 0.1),
    )

    # Create legend for models
    legend2 = plt.legend(
        handles[-3:],
        labels[-3:],
        loc="lower center",
        ncol=len(labels[-3:]),
        bbox_to_anchor=(-0.11, -0.5),
    )

    # Add legends to the figure
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=500)

def plot6in2_1(rs, plot_path):
    site_colors = {
        "DU002": "blue",
        "GN002": "green",
        "KA002": "red",
        "JA002": "purple",
        "MB": "orange",
        "PM": "cyan",
        "QP001": "pink",
    }
    # Define line styles for each model
    model_line_styles = {"ABED": "-", "ECMWF": "--"}
    sns.set(style="whitegrid")
    # Create a 2x2 plot
    fig, axs = plt.subplots(
        4, 2, figsize=(6, 8), gridspec_kw={"height_ratios": [5, 5,5, 2]}
    )

    # Plot MAE_speed
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_speed",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("MAE_speed (km/h)")
    axs[0, 0].set_ylabel("")
    axs[0, 0].set_xlabel("")
    axs[0, 0].legend().set_visible(False)
    axs[0, 0].set_xlim(0, 8)  # Set x-axis limit
    # Plot RMSE_speed
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_speed",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[0,1],
    )
    axs[0,1].set_title("RMSE_speed (km/h)")
    axs[0,1].set_ylabel("")
    axs[0,1].set_xlabel("")
    axs[0,1].legend().set_visible(False)
    axs[0,1].set_xlim(0, 8)  # Set x-axis limit
    # Plot MAE_sine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_sine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[1,0],
    )
    axs[1,0].set_title("MAE_sine")
    axs[1,0].set_ylabel("")
    axs[1,0].set_xlabel("")
    axs[1,0].legend().set_visible(False)
    axs[1,0].set_xlim(0, 8)  # Set x-axis limit
    # Plot RMSE_sine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_sine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[1, 1],
    )
    axs[1, 1].set_title("RMSE_sine")
    axs[1, 1].set_ylabel("")
    axs[1, 1].set_xlabel("")
    axs[1, 1].legend().set_visible(False)
    axs[1, 1].set_xlim(0, 8)  # Set x-axis limit
    # Plot MAE_cosine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_cosine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[2, 0],
    )
    axs[2, 0].set_title("MAE_cosine")
    axs[2, 0].set_ylabel("")
    axs[2, 0].set_xlabel("horizon (hr)")
    axs[2, 0].legend().set_visible(False)
    axs[2, 0].set_xlim(0, 8)  # Set x-axis limit

    # Plot RMSE_cosine
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_cosine",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[2, 1],
    )
    axs[2, 1].set_title("RMSE_cosine")
    axs[2, 1].set_ylabel("")
    axs[2, 1].set_xlabel("horizon (hr)")
    axs[2, 1].legend().set_visible(False)
    axs[2, 1].set_xlim(0, 8)  # Set x-axis limit

    # Hide the last row of subplots
    axs[3, 0].axis("off")
    axs[3, 1].axis("off")

    # Adjust y ticks for right subplots
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    # Adjust layout
    plt.subplots_adjust(hspace=0.2)

    # Create legend for colors
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend1 = plt.legend(
        handles[:-3],
        labels[:-3],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(-0.11, 0.1),
    )

    # Create legend for models
    legend2 = plt.legend(
        handles[-3:],
        labels[-3:],
        loc="lower center",
        ncol=len(labels[-3:]),
        bbox_to_anchor=(-0.11, -0.5),
    )

    # Add legends to the figure
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=500)

def plot6in2_2(rs, plot_path):
    site_colors = {
        "DU002": "blue",
        "GN002": "green",
        "KA002": "red",
        "JA002": "purple",
        "MB": "orange",
        "PM": "cyan",
        "QP001": "pink",
    }
    # Define line styles for each model
    model_line_styles = {"ABED": "-", "ECMWF": "--"}
    sns.set(style="whitegrid")
    # Create a 2x2 plot
    fig, axs = plt.subplots(
        3, 2, figsize=(6, 5), gridspec_kw={"height_ratios": [5, 5, 2]}
    )

    # Plot MAE_u10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_u10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("MAE_u10")
    axs[0, 0].set_ylabel("")
    axs[0, 0].set_xlabel("")
    axs[0, 0].legend().set_visible(False)
    axs[0, 0].set_xlim(0, 8)  # Set x-axis limit

    # Plot RMSE_u10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_u10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("RMSE_u10")
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_xlabel("")
    axs[0, 1].legend().set_visible(False)
    axs[0, 1].set_xlim(0, 8)  # Set x-axis limit

    # Plot MAE_v10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="MAE_v10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[1, 0],
    )
    axs[1, 0].set_title("MAE_v10")
    axs[1, 0].set_ylabel("")
    axs[1, 0].set_xlabel("horizon (hr)")
    axs[1, 0].legend().set_visible(False)
    axs[1, 0].set_xlim(0, 8)  # Set x-axis limit

    # Plot RMSE_v10
    sns.lineplot(
        data=rs,
        x="horizon",
        y="RMSE_v10",
        hue="site",
        style="model",
        palette=site_colors,
        ax=axs[1, 1],
    )
    axs[1, 1].set_title("RMSE_v10")
    axs[1, 1].set_ylabel("")
    axs[1, 1].set_xlabel("horizon (hr)")
    axs[1, 1].legend().set_visible(False)
    axs[1, 1].set_xlim(0, 8)  # Set x-axis limit

    # Hide the last row of subplots
    axs[2, 0].axis("off")
    axs[2, 1].axis("off")

    # Adjust y ticks for right subplots
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    # Adjust layout
    plt.subplots_adjust(hspace=0.2)

    # Create legend for colors
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend1 = plt.legend(
        handles[:-3],
        labels[:-3],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(-0.11, 0.1),
    )

    # Create legend for models
    legend2 = plt.legend(
        handles[-3:],
        labels[-3:],
        loc="lower center",
        ncol=len(labels[-3:]),
        bbox_to_anchor=(-0.11, -0.5),
    )

    # Add legends to the figure
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=500)

def convert_uv_to_speed_deg(u, v):
    speed = np.sqrt(u**2 + v**2)
    deg = np.degrees(np.arctan2(u, v))
    deg = (deg + 360) % 360
    return speed, deg


def plot_abs(file,site, horizons, start_time, end_time,key):
    df = preprocess(file)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    hourly_range = pd.date_range(start=start_time, end=end_time, freq="h")
    # Select rows within the specified time range, but for different horizons
    df_plot = pd.DataFrame()
    for horizon in horizons:
        df_site_hz = df[(df["site"] == site) & (df["horizon"] == horizon)]
        df_site_hz_t = df_site_hz[df_site_hz["time_delta"].isin(hourly_range)]
        for model in ["Ground truth", "ABED", "ECMWF"]:
            df_plot_per = pd.DataFrame()
            df_plot_per["time"] = df_site_hz_t["time_delta"]
            if model == "Ground truth":
                df_plot_per["model"] = model
                df_plot_per["speed"], df_plot_per["deg"] = convert_uv_to_speed_deg(
                    df_site_hz_t["label_u"].values, df_site_hz_t["label_v"].values
                )
            elif model == "ECMWF":
                df_plot_per["model"] = model
                df_plot_per["speed"], df_plot_per["deg"] = convert_uv_to_speed_deg(
                    df_site_hz_t["ecmwf_u"].values, df_site_hz_t["ecmwf_v"].values
                )
            else:
                correct_horizon = 8 if horizon == 7.5 else horizon
                df_plot_per["model"] = f"ABED:hrz-{correct_horizon}hr"
                df_plot_per["speed"], df_plot_per["deg"] = convert_uv_to_speed_deg(
                    df_site_hz_t["pred_u"].values, df_site_hz_t["pred_v"].values
                )
            df_plot = pd.concat([df_plot, df_plot_per])
    df_plot = df_plot.sort_values(by=["time", "model"])

    df_plot = df_plot.drop_duplicates()

    model_colors = {
        "Ground truth": "black",
        "ECMWF": "blue",
        "ABED:hrz-0.5hr": "red",
        "ABED:hrz-1hr": "orange",
        "ABED:hrz-4hr": "blueviolet",
        "ABED:hrz-8hr": "lime",
    }
    sns.set(style="whitegrid")
    sns.lineplot(
        data=df_plot,
        x="time",
        y="speed",
        hue="model",
        linewidth=1,
        alpha=0.8,
        palette=model_colors,
    )
    # Plot arrows for each time stamp
    for idx, row in df_plot.iterrows():
        time = row["time"]
        model = row["model"]
        speed = row["speed"]
        deg = row["deg"]
        color = model_colors[model]
        # time = time.strftime("%m-%d %Hh")
        # Plot arrow
        plt.quiver(
            time,
            speed,
            np.sin(np.deg2rad(deg)),
            np.cos(np.deg2rad(deg)),
            color=color,
            alpha=0.7,
            scale=10,
            scale_units="inches",
            width=0.007,
            headwidth=2,
            headlength=3.0,
            headaxislength=3.0,
        )
        # plt.text(time, speed, model, ha='center', va='bottom', color=color)

    # Set x-axis limits and labels
    plt.xlim(df_plot["time"].min(), df_plot["time"].max())
    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter("%I%p")
    )
    plt.xlabel("Time (UTC+0H)")

    # Set y-axis label and limits
    plt.ylim(0, df_plot["speed"].max() * 1.1)
    plt.ylabel("Speed (km/h)")
    plt.legend(fontsize='small')
    # Set title
    # plt.title(f'{site} Wind Forecast: {8 if horizon==7.5 else horizon}-Hour Horizon\n({start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")})')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        f'./abs_{site}_h{horizons}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}_{key}.png',
        dpi=500,
    )
    plt.close()


def plot_r_dem(file,hr,key):
    horizon = pd.to_timedelta(hr,unit='h')
    with rasterio.open(dem_data_path) as dem:
        dem_data = dem.read(1)  # Read the first band
    dem_data[dem_data < 0] = np.nan

    RECT_LAT_MIN = -36
    RECT_LAT_MAX = -24.8
    RECT_LON_MIN = 113
    RECT_LON_MAX = 127.4

    PATCH_LAT_MIN = -32 - 3.4
    PATCH_LAT_MAX = -32.0
    PATCH_LON_MIN = 115.0
    PATCH_LON_MAX = 115 + 3.4

    def clip(key):
        dem_lat_max = -9.00  # - 0.0025/2
        dem_lat_min = -43.7425  # + 0.0025/2
        dem_lon_min = 112.90  # + 0.0025/2
        dem_lon_max = 154.00  # - 0.0025/2
        rectgrid_lat = np.linspace(
            dem_lat_max, dem_lat_min, num=dem_data.shape[0], endpoint=True, dtype="float64"
        )
        rectgrid_lon = np.linspace(
            dem_lon_min, dem_lon_max, num=dem_data.shape[1], endpoint=True, dtype="float64"
        )
        if key == "patch":
            lat_min_i = np.argmin(np.abs(rectgrid_lat - PATCH_LAT_MIN))
            lat_max_i = np.argmin(np.abs(rectgrid_lat - PATCH_LAT_MAX))
            lon_min_i = np.argmin(np.abs(rectgrid_lon - PATCH_LON_MIN))
            lon_max_i = np.argmin(np.abs(rectgrid_lon - PATCH_LON_MAX))
        else:
            lat_min_i = np.argmin(np.abs(rectgrid_lat - RECT_LAT_MIN))
            lat_max_i = np.argmin(np.abs(rectgrid_lat - RECT_LAT_MAX))
            lon_min_i = np.argmin(np.abs(rectgrid_lon - RECT_LON_MIN))
            lon_max_i = np.argmin(np.abs(rectgrid_lon - RECT_LON_MAX))
        terrain = dem_data[lat_max_i:lat_min_i, lon_min_i:lon_max_i]
        return terrain

    terrain = clip("patch")
    terrain = masked_array(terrain, np.isnan(terrain))

    def r_color(value):
        clr = plt.cm.coolwarm((value + 1) / 2)
        return clr

    fig, axes = plt.subplots(2, 2, figsize=(6, 6),sharex=True,sharey=True)

    # Loop through each subplot
    for i, ax in enumerate(axes.flat):
        # Plot terrain as background
        img_terrain = ax.imshow(
            terrain,
            cmap="terrain",
            extent=[PATCH_LON_MIN, PATCH_LON_MAX, PATCH_LAT_MIN, PATCH_LAT_MAX],
            alpha=0.3
        )
        # Add colorbar at the bottom of the grid
        # if i == 3:  # For the last subplot
        #     cbar = plt.colorbar(ax=ax, orientation='horizontal', label="Elevation (m)")
        # Add labels
        ax.set_xlabel("Longitude ({} to {})".format(PATCH_LON_MIN, PATCH_LON_MAX))
        ax.set_ylabel("Latitude ({} to {})".format(PATCH_LAT_MIN, PATCH_LAT_MAX))

    file['horizon'] = pd.to_timedelta(file['horizon'])
    file = file[file['horizon']==horizon]
    sites = file['site'].unique()

    for ix, row in ALL_STATION_COORDS.iterrows():
        if row['name'] not in sites:
            continue
        r_u = file[file['site']==row['name']]['r_u']
        r_v = file[file['site']==row['name']]['r_v']
        ecmwf_r_u = file[file['site']==row['name']]['ecmwf_r_u']
        ecmwf_r_v = file[file['site']==row['name']]['ecmwf_r_v']

        axes[0][0].scatter(
            row["lon"], row["lat"], color=r_color(r_u), s=15
        )
        axes[0][1].scatter(
            row["lon"], row["lat"], color=r_color(r_v), s=15
        )
        axes[1][0].scatter(
            row["lon"], row["lat"], color=r_color(ecmwf_r_u), s=15
        )
        axes[1][1].scatter(
            row["lon"], row["lat"], color=r_color(ecmwf_r_v), s=15
        )
    axes[0][0].set_title(f"A - r(u10_ABED, u3)")
    axes[0][1].set_title(f"B - r(v10_ABED, v3)")
    axes[1][0].set_title(f"C - r(u10_ECMWF, u3)")
    axes[1][1].set_title(f"D - r(v10_ECMWF, v3)")

    for ix, row in STAR_STATION_CORRDS.iterrows():
        if row['name'] not in sites:
            continue
        axes[0][0].scatter(
            row["lon"], row["lat"], color="green", s=20, marker="o",  # Use a circle marker
            facecolors='none',  # Make the marker face color transparent
            edgecolors='blue',  # Set the marker edge color to red
            linewidth=0.8,  # Set the width of the marker edge
        )  # Latitude is y, Longitude is x
        axes[0][1].scatter(
            row["lon"], row["lat"], color="green", s=20, marker="o",  # Use a circle marker
            facecolors='none',  # Make the marker face color transparent
            edgecolors='blue',  # Set the marker edge color to red
            linewidth=0.8,  # Set the width of the marker edge
        )
        axes[1][0].scatter(
            row["lon"], row["lat"], color="green", s=20, marker="o",  # Use a circle marker
            facecolors='none',  # Make the marker face color transparent
            edgecolors='blue',  # Set the marker edge color to red
            linewidth=0.8,  # Set the width of the marker edge
        )
        axes[1][1].scatter(
            row["lon"], row["lat"], color="green", s=20, marker="o",  # Use a circle marker
            facecolors='none',  # Make the marker face color transparent
            edgecolors='blue',  # Set the marker edge color to red
            linewidth=0.8,  # Set the width of the marker edge,
        )
    plt.tight_layout()

    cbar_terrain_ax = fig.add_axes([0.1, 0.08, 0.4, 0.02])  # Position and size of terrain colorbar
    cbar_corr_ax = fig.add_axes([0.55, 0.08, 0.4, 0.02])
    cbar_terrain = plt.colorbar(img_terrain,cax=cbar_terrain_ax,orientation='horizontal',
                                label="Elevation (m)",pad=0.05,shrink=0.9,aspect=40)

    cbar_cor = plt.colorbar(plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1)),
                            cax=cbar_corr_ax,orientation='horizontal',
                            label="Correlation",shrink=0.9,aspect=40)

    plt.subplots_adjust(bottom=0.18)
    # plt.tight_layout()

    plt.savefig(f"r_hrz{hr}_{key}.png", dpi=500)


command1 = "mae_rmse"
command2 = "abs"
command3 = "r_3m"
commands = [command1]
if __name__ == "__main__":
    keys = ['loss2']
    for key in keys:
        starfile,station3mfile = files(key)
        for command in commands:
            if command == command1:
                for type in ["all", "winternight", "winterday", "summernight", "summerday"]:
                    rspath = f"./7sites_{type}_{key}.csv"
                    plot1_path = f"./7sites_{type}_{key}_spdeg.png"
                    plot2_path = f"./7sites_{type}_{key}_uv.png"
                    plot_path = f"./7site_{type}_{key}.png"
                    rs = get_rs(rspath,starfile, type,key=key)
                    plot6in1(rs, plot_path)
                    plot6in2_1(rs,plot1_path)
                    plot6in2_2(rs,plot2_path)

            if command == command2:
                sites = ["GN002","PM","QP001","DU002","KA002","JA002","MB"]
                horizons = [0.5, 7.5]
                start_times = ["2022-07-29 00:00:00+00:00", "2023-01-29 00:00:00+00:00"]
                end_times = ["2022-07-31 23:45:00+00:00", "2023-01-31 23:45:00+00:00"]
                for site in sites:
                    for start_time, end_time in zip(start_times, end_times):
                        plot_abs(starfile,site, horizons, start_time, end_time,key)

            if command ==command3:
                file = pd.read_csv(station3mfile)
                plot_r_dem(file,hr=0.5,key=key)
                plot_r_dem(file,hr=2,key=key)
                plot_r_dem(file,hr=4,key=key)
                plot_r_dem(file,hr=7.5,key=key)