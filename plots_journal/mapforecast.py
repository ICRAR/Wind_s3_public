import os.path
from PIL import Image
import pandas as pd
import xarray as xr
import numpy as np
from metrics_outfile.collect_result import convert_uv_to_vdir
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.colors as mcolors


data_path = ("/mnt/science1/fchen/result_ST/lat32_lon115_3d4_2022_2023/"
             "35.4_32.0_115.0_118.4_7feat_terrain_2022_2023_loss2_predst_2022_7_1_2022_10_1_ds.nc")
v_max = 45
save_path = "./2022_07"
data = xr.open_dataset(data_path)
data = data.sel(time_since=slice('2022-07-01', '2022-08-01'))


# data_path = ("/mnt/science1/fchen/result_ST/lat32_lon115_3d4_2022_2023/"
#              "35.4_32.0_115.0_118.4_7feat_terrain_2022_2023_loss2_predst_2023_1_1_2023_4_1_ds.nc")
# v_max = 35
# save_path = "./2023_01"
# data = xr.open_dataset(data_path)
# data = data.sel(time_since=slice('2023-01-01', '2023-02-01'))

hrz = "30min"

polygon_path = "../../../../data_prep_ST/make_grid/prediction_area.shp"

data["time_to"] = pd.to_datetime(data["time_to"])
T_hr = 48
L_hr = 4
F_hr = 4
n_pred = (F_hr + L_hr) * 4 + 1
lat_min = data["latitude"].min().values
lat_max = data["latitude"].max().values
lon_min = data["longitude"].min().values
lon_max = data["longitude"].max().values
v_min = 0

if os.path.exists(save_path):
    files = os.listdir(save_path)
    for file in files:
        os.remove(os.path.join(save_path, file))
else:
    os.makedirs(save_path)


def plot(data_frame):
    def coord():
        x, y = data_frame["longitude"], data_frame["latitude"]
        x_flat, y_flat = np.meshgrid(x, y)
        x_flat = x_flat.flatten()
        y_flat = y_flat.flatten()
        return x_flat, y_flat

    def plot_arrow(ax, deg_flat: np.ndarray):
        length = 0.6 * 0.11
        x_flat, y_flat = coord()
        for x, y, deg in zip(x_flat, y_flat, deg_flat):
            x = x + length / 2 * np.cos(np.radians(deg) - np.pi / 2)
            y = y - length / 2 * np.sin(np.radians(deg) - np.pi / 2)
            dx = -length * np.cos(np.radians(deg) - np.pi / 2)
            dy = length * np.sin(np.radians(deg) - np.pi / 2)
            ax.arrow(
                x, y, dx, dy, width=0.000001, length_includes_head=True, head_width=0.02
            )

    def plot_color_arrow(ax, deg_flat: np.ndarray, speed_flat: np.ndarray,cm):
        length = 1 * 0.11
        x_flat, y_flat = coord()
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)  # Set vmin and vmax according to your data
        # Get color from colormap

        for x, y, deg, speed in zip(x_flat, y_flat, deg_flat,speed_flat):
            x = x + length / 2 * np.cos(np.radians(deg) - np.pi / 2)
            y = y - length / 2 * np.sin(np.radians(deg) - np.pi / 2)
            dx = -length * np.cos(np.radians(deg) - np.pi / 2)
            dy = length * np.sin(np.radians(deg) - np.pi / 2)
            ax.arrow(
                x, y, dx, dy, facecolor = cm(norm(speed)), edgecolor = 'black', linewidth = 1,
                width=0.01, length_includes_head=True, head_width=0.06
            )

    polygon = gpd.read_file(polygon_path)

    data_frame["time_to"] = data_frame["time_to"]+pd.to_timedelta('8H')
    time = np.datetime_as_string(data_frame["time_to"].values.astype("datetime64[m]"))
    cm = matplotlib.colormaps["rainbow"]

    # 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu','RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'nipy_spectral'
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[10, 0, 0.5])
    # Subplots
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    cax = plt.subplot(gs[2, :])
    image0 = ax0.imshow(
        data_frame["pred_speed"],
        extent=(lon_min, lon_max, lat_min, lat_max),
        cmap=cm,
        vmin=v_min,
        vmax=v_max,
    )
    image1 = ax1.imshow(
        data_frame["ecmwf_speed"],
        extent=(lon_min, lon_max, lat_min, lat_max),
        cmap=cm,
        vmin=v_min,
        vmax=v_max,
    )

    plot_arrow(ax0, data_frame["pred_deg"].values.flatten())
    plot_arrow(ax1, data_frame["ecmwf_deg"].values.flatten())

    label_speed_flat = data_frame["label_speed"].values.flatten()
    label_deg_flat = data_frame["label_deg"].values.flatten()
    plot_color_arrow(ax0,label_deg_flat,label_speed_flat,cm)
    plot_color_arrow(ax1,label_deg_flat,label_speed_flat,cm)
    # mask = ~np.isnan(label_speed_flat)
    # x_speed, y_speed = coord()
    # ax0.scatter(
    #     x_speed[mask],
    #     y_speed[mask],
    #     c=label_speed_flat[mask],
    #     cmap=cm,
    #     marker="o",
    #     vmin=v_min,
    #     vmax=v_max,
    #     edgecolors="black",
    # )
    # ax1.scatter(
    #     x_speed[mask],
    #     y_speed[mask],
    #     c=label_speed_flat[mask],
    #     cmap=cm,
    #     marker="o",
    #     vmin=v_min,
    #     vmax=v_max,
    #     edgecolors="black",
    # )

    polygon.boundary.plot(ax=ax0, color="black", linewidth=1)
    polygon.boundary.plot(ax=ax1, color="black", linewidth=1)

    ax0.set_xlabel("Longitude", fontsize=14, fontweight="bold")
    ax0.set_ylabel("Latitude", fontsize=14, fontweight="bold")
    ax0.set_xlim(lon_min, lon_max)
    ax0.set_ylim(lat_min, lat_max)
    ax0.set_title(f"Model prediction on {time} (UTC+8H)", fontweight="bold")

    ax1.set_xlabel("Longitude", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Latitude", fontsize=14, fontweight="bold")
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)
    ax1.set_title(f"ECMWF forecast on {time} (UTC+8H)", fontweight="bold")

    # Add colorbar
    colorbar = plt.colorbar(image0, cax=cax, orientation="horizontal")
    colorbar.set_label("Wind speed (km/h)", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{time}"), dpi=100)
    plt.close()


def create_gif(path):
    image_files = [os.path.join(path, f) for f in os.listdir(path)][::2]
    image_files.sort()
    file_name = data_path.replace('_ds',f"hrz{hrz}") + "_gif.gif"
    images = []
    for image_file in image_files:
        img = Image.open(image_file)
        images.append(img)
    images[0].save(
        file_name, save_all=True, append_images=images[1:], duration=250, loop=0
    )


for since in data["time_since"]:
    to = since + pd.to_timedelta(hrz)
    data_frame = data.sel(time_since=since, time_to=to)
    data_frame["pred_speed"], data_frame["pred_deg"], *_ = convert_uv_to_vdir(
        data_frame["pred_u"], data_frame["pred_v"]
    )
    data_frame["ecmwf_speed"], data_frame["ecmwf_deg"], *_ = convert_uv_to_vdir(
        data_frame["ecmwf_u"], data_frame["ecmwf_v"]
    )
    data_frame["label_speed"], data_frame["label_deg"], *_ = convert_uv_to_vdir(
        data_frame["label_u"], data_frame["label_v"]
    )
    print(str(since.values))
    plot(data_frame)

create_gif(save_path)
# shutil.rmtree(save_path)
