import xarray as xr
import numpy as np
import pandas as pd
import os
import sys

sys.path.extend(["/home/fchen/wind"])
from data_prep_ST.preprocess import preprocess

print(sys.path)


# cds
def create_cds_dataset(args):
    if os.path.exists(args.cds_dest_path) == False:
        print("The cds gridded data is not available, creating from scratch")
        cds = xr.load_dataset(args.cds_src_path)
        cds = preprocess.uniform_cds(cds, args.start_cds, args.end)
        cds_space = preprocess.interp_space(
            cds,
            interp_mode="linear",
            grid_deg=args.grid_deg,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
        )
        cds_time = preprocess.interp_time(
            cds_space, "15T", args.vars_cds, args.start_dpird, args.end
        )
        cds_time.to_netcdf(args.cds_dest_path)


def create_dpird_dataset(args):
    if os.path.exists(args.dpird_dest_path) == False:
        print("The dpird gridded data is not available, creating from scratch")
        dataset = preprocess.form_dpird_dataset(
            args.dpird_src_path,
            args.stations_coords,
            grid_deg=args.grid_deg,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            start=args.start_cds,
            end=args.end,
        )
        dataset.to_netcdf(args.dpird_dest_path)


def create_terrain_dataset(args):
    if os.path.exists(args.terrain_dest_path) == False:
        print("The terrain gridded data is not available, creating from scratch")
        terrain = preprocess.create_terrain_grid(
            args.terrain_src_path,
            grid_deg=args.grid_deg,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            start=args.start_cds,
            end=args.end,
        )
        terrain.to_netcdf(args.terrain_dest_path)


def create_label_dataset(args):
    if os.path.exists(args.label_dest_path) == False:
        print("The label gridded data is not available, creating from scratch")
        grid_label = preprocess.create_label_grid(
            args.dpird_src_path,
            args.stars_coords,
            grid_deg=args.grid_deg,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            start=args.start_cds,
            end=args.end,
        )
        grid_label.to_netcdf(args.label_dest_path)


def create_label3m_dataset(args):
    if os.path.exists(args.label3m_dest_path) == False:
        print("The label3m gridded data is not available, creating from scratch")
        grid_label = preprocess.create_label3m_grid(
            args.dpird_src_path,
            args.stations_coords,
            grid_deg=args.grid_deg,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            start=args.start_cds,
            end=args.end,
        )
        grid_label.to_netcdf(args.label3m_dest_path)
