import argparse
import os
from exp.exp import Exp
import mlflow
import mlflow.pytorch
import json

# from exp.exp_st_main_dist import Exp as Exp_dist
# from exp.exp_st_main import Exp as Exp_solo
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--gpu_ix", type=int, default=1)
parser.add_argument(
    "--flag",
    type=str,
    default="test_grid",
    help="need to be among train, test_star, test_grid",
)
parser.add_argument("--mlflow", action="store_true")

parser.add_argument("--backend", type=str, default="nccl")
parser.add_argument("--init_method", type=str, default="env://")
parser.add_argument("--customise_grid", action="store_true",default=True)
parser.add_argument(
    "--customise_grid_config",
    nargs="+",
    type=float,
    default=[-35.4, -32.0, 115.0, 118.4],
    help="[the NWest lat, lon, degrees to the East,South ]",
)
parser.add_argument("--grid_ix", type=int, default=4)
parser.add_argument("--grid_file", type=str, default="./utils/grids.csv")
parser.add_argument("--grid_deg", type=float, default=0.1)
parser.add_argument("--start_cds", type=str, default="2022-1-1 00:00:00")
parser.add_argument("--start_dpird", type=str, default="2022-1-1 00:00:00")
parser.add_argument(
    "--end", type=str, default="2023-12-31 23:45:00"
)  # '2023-10-1 00:00:00'
parser.add_argument("--test_grid_start", type=str, default='2022-07-01 00:00:00')
parser.add_argument("--test_grid_end", type=str, default='2022-08-01 00:00:00')
parser.add_argument("--vars_terrain", action="store_true",default=True)
parser.add_argument("--vars_cds", nargs="+", type=str, default=["u10", "v10", "msl"])
parser.add_argument(
    "--vars_dpird",
    nargs="+",
    type=str,
    default=["wind_3m_u", "wind_3m_v", "airTemperature", "relativeHumidity"],
)
parser.add_argument(
    "--labels",
    nargs="+",
    type=str,
    default=[
        "wind_10m_u",
        "wind_10m_v",
    ],  # wind_10m_u and wind_10m_v must be the first 2 items
)

parser.add_argument("--test_n_days", type=int, default=5)
parser.add_argument("--T_hr", type=int, default=48, help="sample length")  # 48
parser.add_argument(
    "--L_hr", type=int, default=4, help="moving window of y from x"
)  # 4
parser.add_argument(
    "--S_min", type=int, default=15, help="sliding window of each sample"
)  # 15
parser.add_argument(
    "--F_hr", type=int, default=4, help="forecasting of ecmwf, F is added on top of T"
)  # 4
parser.add_argument("--filters", nargs="+", type=int, default=[4, 8, 16])  # [4, 8, 16]

parser.add_argument(
    "--datasrc",
    type=int,
    default=2,
    help="0-dpird only, 1-ecmwf only, 2-dpird and ecmwf",
)

parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)  # 32
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument(
    "--stations_coords",
    type=str,
    default="./data_prep_ST/make_grid/all_station_coordinates.csv",
)
parser.add_argument(
    "--stars_coords",
    type=str,
    default="./data_prep_ST/make_grid/stations_available_label_coordinates.csv",
)
parser.add_argument(
    "--cds_src_path", type=str, default="/mnt/science1/fchen/dataset_CDS_2019_2023/combined_p01_2019_2023.nc"
)
parser.add_argument(
    "--terrain_src_path",
    type=str,
    default="/mnt/science1/fchen/dataset_terrain/dem-9s.tif",
)
parser.add_argument("--dpird_src_path", type=str, default="/mnt/science1/fchen/dataset_DPIRD")
parser.add_argument("--st_data_path", type=str, default="/mnt/science1/fchen/dataset_ST/")
parser.add_argument(
    "--result_path",
    type=str,
    default="/mnt/science1/fchen/result_ST/lat32_lon115_3d4_2022_2023/",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="/mnt/science1/fchen/model_ST/",
)

args = parser.parse_args()


def main():
    if args.flag == "train":
        if args.distributed:
            print("-" * 30, "IT IS SET IN DISTRIBUTED MODE", "-" * 30)
            if args.init_method == "env://":
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "2323"
            mp.set_start_method("spawn", force=True)
            world_size = args.gpu_ix
            mp.spawn(
                main_worker, nprocs=args.gpu_ix, args=(world_size, args), join=True
            )

        else:
            print("-" * 30, "NORMAL MODE", "-" * 30)
            exp = Exp(args)
            exp.train()
    elif args.flag == "test_star":
        args.distributed = False
        print("-" * 30, "NORMAL MODE", "-" * 30)
        exp = Exp(args)
        exp.test_star()
    elif args.flag == "test_grid":
        args.distributed = False
        print("-" * 30, "NORMAL MODE", "-" * 30)
        exp = Exp(args)
        exp.test_grid()


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        world_size=world_size,
        rank=rank,
    )
    print(f"initiatialized distribution for {rank} out of {world_size}")
    exp = Exp(args, rank, world_size)
    exp.train()


if __name__ == "__main__":
    if args.mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        logname = (
            f"T{args.T_hr}_L{args.L_hr}_F{args.F_hr}_S{args.S_min}_"
            f'flt{args.filters[0]}_{args.filters[1]}_{args.filters[2]}_{"_".join(map(str, args.vars_dpird+args.vars_cds))}'
        )
        if args.flag == "train":
            mlflow.set_experiment("Wind_ST")
            with mlflow.start_run(run_name=logname) as run:
                # Log hyperparameters
                mlflow.set_tag("logname", logname)
                mlflow.log_param(
                    "features", "_".join(map(str, args.vars_dpird + args.vars_cds))
                )
                mlflow.log_param("T_hr", args.T_hr)
                mlflow.log_param("L_hr", args.L_hr)
                mlflow.log_param("F_hr", args.F_hr)
                mlflow.log_param("S_min", args.S_min)
                mlflow.log_param("filters", "_".join(map(str, args.filters)))
                mlflow.log_param(
                    "customise_grid", "_".join(map(str, args.customise_grid_config))
                )
                main()
            mlflow.end_run()
        else:  # for test_star and test_grid
            filter_string = f"tags.logname = '{logname}'"
            df = mlflow.search_runs(
                experiment_names=["Wind_ST"], filter_string=filter_string
            )
            args.run_id = df["run_id"].values.tolist()[0]
            print(args.run_id)
            with mlflow.start_run(run_name=logname) as run:
                main()
            mlflow.end_run()

    else:
        main()
