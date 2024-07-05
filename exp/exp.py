import pandas as pd
import numpy as np
import os, time, psutil
import xarray as xr
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from data_prep_ST.preprocess import p1_grid_data, p2_data_generator
from models import ABED
from loss.MSE import MSE_point, MSE_10_3_point
from metrics_outfile import out_site
from utils.earlystop import EarlyStopping
import matplotlib.pyplot as plt
import mlflow.pytorch
from metrics_outfile.collect_result import generate_result_csv,generate_correlation_csv


class Exp(object):
    def __init__(self, args, rank=None, world_size=None):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.args.grid_deg = 0.1
        # self.device = torch.device("cuda")

        if self.args.customise_grid:
            self.args.lat_min = self.args.customise_grid_config[0]
            self.args.lat_max = self.args.customise_grid_config[1]
            self.args.lon_min = self.args.customise_grid_config[2]
            self.args.lon_max = self.args.customise_grid_config[3]
        else:
            grid_df = pd.read_csv(self.args.grid_file)
            info = grid_df[grid_df["ix"] == self.args.grid_ix]
            self.args.lat_min = info["lat_min"].values[0]
            self.args.lat_max = info["lat_max"].values[0]
            self.args.lon_min = info["lon_min"].values[0]
            self.args.lon_max = info["lon_max"].values[0]

        print(
            f"The map - lat_min {self.args.lat_min}, lat_max{self.args.lat_max}, \
            lon_min{self.args.lon_min}, lon_max{self.args.lon_max}"
        )
        if self.args.datasrc not in [0, 1, 2]:
            raise ValueError(
                "data source must be 0 (dpird only), or 1 (ecmwf only), or 2 (dpird and ecmwf)."
            )
        if self.args.customise_grid:
            self.args.filename = (
                f"{-self.args.lat_min}_{-self.args.lat_max}_"
                f"{self.args.lon_min}_{self.args.lon_max}_"
                f"{len(self.args.vars_cds + self.args.vars_dpird)}feat_"
                f"{'terrain' if self.args.vars_terrain else ''}_2022_2023_loss{len(self.args.labels)}"
            )
        else:
            self.args.filename = (
                f"ix{self.args.grid_ix}_{-self.args.lat_min}_"
                f"{-self.args.lat_max}_{self.args.lon_min}_"
                f"{self.args.lon_max}_{len(self.args.vars_cds + self.args.vars_dpird)}"
                f"feat_2022_2023_loss{len(self.args.labels)}")
        self.args.cds_dest_path = os.path.join(
            self.args.st_data_path, "cds_" + self.args.filename + ".nc"
        )
        self.args.dpird_dest_path = os.path.join(
            self.args.st_data_path, "dpird_" + self.args.filename + ".nc"
        )
        self.args.terrain_dest_path = os.path.join(
            self.args.st_data_path, "terrain_" + self.args.filename + ".nc"
        )
        self.args.label_dest_path = os.path.join(
            self.args.st_data_path, "label_" + self.args.filename + ".nc"
        )
        self.args.label3m_dest_path = os.path.join(
            self.args.st_data_path, "label3m_" + self.args.filename + ".nc"
        )
        self.args.train_path = os.path.join(
            self.args.st_data_path, "train_" + self.args.filename + ".nc"
        )
        self.args.test_path = os.path.join(
            self.args.st_data_path, "test_" + self.args.filename + ".nc"
        )
        self.args.test_noscale_path = os.path.join(
            self.args.st_data_path, "test_noscale_" + self.args.filename + ".nc"
        )

        self.args.loss_save_path = os.path.join(
            self.args.result_path, "loss_" + self.args.filename + ".png"
        )
        self.args.result_save_path = os.path.join(
            self.args.result_path, self.args.filename + "_predstar.csv"
        )
        self.args.result3m_save_path = os.path.join(
            self.args.result_path, f"{self.args.filename}_predst_{self.args.test_grid_start.replace('-', '_').replace('.', '_')}"
                                   f"_{self.args.test_grid_end.replace('-', '_').replace('.', '_')}_predstation3m.csv"
        )
        self.args.ds_save_path = os.path.join(
            self.args.result_path,
            f"{self.args.filename}_predst_{self.args.test_grid_start.replace('-', '_').replace('.', '_')}"
            f"_{self.args.test_grid_end.replace('-', '_').replace('.', '_')}_ds.nc",
        )

        self.args.model_save_path = os.path.join(
            self.args.model_path, self.args.filename + ".pth"
        )
        self.args.star_coord_ix = os.path.join(
            self.args.result_path, f"star_loc_{self.args.filename}.csv"
        )
        self.args.station3m_coord_ix = os.path.join(self.args.result_path,f"station3m_loc_{self.args.filename}.csv")

        if not os.path.exists(self.args.result_path):
            raise ValueError(f"Please mk directory {self.args.result_path}")
        if not os.path.exists(self.args.model_path):
            raise ValueError(f"Please mk directory {self.args.model_path}")
        if args.distributed:
            # Initialize distributed training
            torch.cuda.set_device(self.rank)
            torch.cuda.empty_cache()
            self.device = torch.device("cuda:{}".format(self.rank))
            print("Using GPU", self.device)
        else:
            torch.cuda.empty_cache()
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                available_gpus = torch.cuda.device_count()
                # Use all available GPUs if gpu_ix is greater than the available GPUs
                self.gpu_ids = (
                    list(range(available_gpus))
                    if self.args.gpu_ix > available_gpus
                    else list(range(self.args.gpu_ix))
                )

                self.device = torch.device("cuda:{}".format(self.gpu_ids[0]))
                # Print the selected GPU(s)
                print("Using GPU(s): {}".format(self.gpu_ids))
            else:
                self.gpu_ids = None
                self.device = torch.device("cpu")
                print("CPU")

        self._create_dataset()

    def _create_dataset(self):
        print(
            "IF IT IS RUN FOR THE FIRST TIME, OR THE AREA HAS BEEN CHANGED, PLEASE DELETE THE TRAINING AND TESTING "
            "FILES."
        )
        if not os.path.exists(self.args.terrain_dest_path):
            p1_grid_data.create_terrain_dataset(self.args)
        if not os.path.exists(self.args.label_dest_path):
            p1_grid_data.create_label_dataset(self.args)
        if not os.path.exists(self.args.label3m_dest_path):
            p1_grid_data.create_label3m_dataset(self.args)
        if not os.path.exists(self.args.dpird_dest_path):
            p1_grid_data.create_dpird_dataset(self.args)
        if not os.path.exists(self.args.cds_dest_path):
            p1_grid_data.create_cds_dataset(self.args)

        if not os.path.exists(self.args.train_path) and not os.path.exists(
            self.args.test_path
        ):
            # create train and test, and also reorganise all the data,
            # including training, testing if customise_grid
            p2_data_generator.Train_Test(self.args).main_generate_dataset()

    def _get_data(self, flag):
        # data loader and sampler if distributed
        dataloader, sampler = p2_data_generator.data_provider(
            self.args, flag=flag, rank=self.rank
        )

        # get data loader shape
        for x, *_ in dataloader:
            _, self.len_feat, self.len_time, self.len_lat, self.len_lon = x.shape
            break
        print(
            f"The dataset is composed of {self.len_feat} features, {self.len_time} time stamps, {self.len_lat} latitudes, and {self.len_lon} longitudes."
        )

        return dataloader, sampler

    def _build_model(self, n_feat, n_time, n_lat=160, n_lon=184, filters=[16, 32, 64]):
        model = ABED.ABED(
            size_input=(n_time, n_lat, n_lon),
            activation="ReLU",
            channel_input=n_feat,
            channel_output=2,
            filters=filters,
        )
        if self.args.distributed:
            model = model.cuda(self.rank)
            model = DDP(model, device_ids=[self.rank])
        else:
            model = model.to(self.device)

        return model

    def train(self):
        train_dataloader, train_sampler = self._get_data(flag="train")
        print(
            "train_dataloader will have {} samples, which will generates {} batches".format(
                len(train_dataloader.dataset),
                len(train_dataloader.dataset) // self.args.batch_size,
            )
        )

        model = self._build_model(
            n_feat=self.len_feat,
            n_time=self.len_time,
            n_lat=self.len_lat,
            n_lon=self.len_lon,
            filters=self.args.filters,
        )
        if "wind_3m_u" in self.args.labels:
            terrain = torch.tensor(xr.open_dataset(self.args.test_noscale_path)["terrain"][0].data,dtype=torch.float32).to(self.device)
            criterion = MSE_10_3_point(terrain).to(self.device)
        else:
            criterion = MSE_point().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.1, verbose=True
        )
        early_stopping = EarlyStopping(patience=5, verbose=True)

        model.train()
        num_epochs = self.args.epochs
        train_losses = []
        best_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            if self.args.distributed:
                train_sampler.set_epoch(epoch)
            avg_loss = 0.0
            counter = 0

            with tqdm(
                total=len(train_dataloader),
                desc=f"ix{self.args.grid_ix} Epoch {epoch}/{num_epochs}",
                unit="batch",
            ) as pbar:
                for x, y, _, y3m in train_dataloader:
                    # print("In training dataloader, x size is", x.shape)
                    # print("The data is loaded to self.device", self.device)
                    counter += 1
                    model.zero_grad()
                    x = x.to(self.device, non_blocking=True)
                    # print("rank {} x shape".format(self.rank), x.shape)
                    y = y.to(self.device, non_blocking=True)
                    y3m = y3m.to(self.device, non_blocking=True)
                    # print("rank {} y shape".format(self.rank), y.shape)
                    output = model(x.float())

                    if "wind_3m_u" in self.args.labels:
                        loss = criterion(output, y.float(), y3m.float())
                    else:
                        loss = criterion(output, y.float())
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()

                    epoch_loss = avg_loss / counter
                    train_losses.append(epoch_loss)
                    if self.args.mlflow:
                        mlflow.log_metric("train_loss", epoch_loss)
                    pbar.set_postfix({"Loss": epoch_loss})
                    pbar.update()
                    print("\n")
                    # print_gpu_memory_summary(np.arange(self.world_size))

            scheduler.step(epoch_loss)

            # Save the model only if the current loss is the best so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), self.args.model_save_path)

            # Check if early stopping criteria are met
            if early_stopping(epoch_loss):
                print("Early stopping")
                break

        if self.args.distributed:
            dist.destroy_process_group()

        # save the loss plot
        fig, ax = plt.subplots()
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        if self.args.mlflow:
            mlflow.log_figure(fig, self.args.loss_save_path)
        plt.savefig(self.args.loss_save_path)

    def test_star(self):
        test_dataloader, test_sampler = self._get_data(flag="test_star")

        model = self._build_model(
            n_feat=self.len_feat,
            n_time=self.len_time,
            n_lat=self.len_lat,
            n_lon=self.len_lon,
            filters=self.args.filters,
        )

        print("model path", self.args.model_save_path)

        model.load_state_dict(torch.load(self.args.model_save_path))

        model.eval()

        with torch.no_grad(), tqdm(
            total=len(test_dataloader), desc="Testing", unit="batch"
        ) as pbar:
            df = pd.DataFrame()

            for i, (x, y, y_time, _) in enumerate(test_dataloader):
                x = x.to(self.rank) if self.args.distributed else x.to(self.device)
                yhat = model(x.float())
                out = (
                    yhat.to(self.rank)
                    if self.args.distributed
                    else yhat.to(self.device)
                )
                y = y.to(self.rank) if self.args.distributed else y.to(self.device)

                y = y.cpu().numpy()
                out = out.cpu().numpy()

                y_time = y_time.numpy()

                out_df = out_site.out_per_df(self.args, out, y, y_time)
                df = pd.concat([df, out_df])
                pbar.update(1)  # Increment the progress bar

        # store the results
        df.to_csv(self.args.result_save_path, index=False)
        generate_result_csv(self.args)
        if self.args.mlflow:
            mlflow.log_artifact(self.args.result_save_path)

        print("The job is done", "-" * 30)
        return df

    def test_grid(self):
        start_time = time.time()
        test_dataloader, test_sampler = self._get_data(flag="test_grid")

        model = self._build_model(
            n_feat=self.len_feat,
            n_time=self.len_time,
            n_lat=self.len_lat,
            n_lon=self.len_lon,
            filters=self.args.filters,
        )

        print("model path", self.args.model_save_path)

        model.load_state_dict(torch.load(self.args.model_save_path))

        model.eval()

        with torch.no_grad(), tqdm(
            total=len(test_dataloader), desc="Testing", unit="batch"
        ) as pbar:
            ds = []
            df = pd.DataFrame()
            for i, (x, y, ytime, y3m) in enumerate(test_dataloader):
                x = x.to(self.rank) if self.args.distributed else x.to(self.device)
                yhat = model(x.float())
                out = (
                    yhat.to(self.rank)
                    if self.args.distributed
                    else yhat.to(self.device)
                )
                y = y.to(self.rank) if self.args.distributed else y.to(self.device)

                y = y.cpu().numpy()
                out = out.cpu().numpy()
                ytime = ytime.numpy()

                y3m = y3m.cpu().numpy()

                out_ds = out_site.out_ds(self.args, out, y, ytime)
                ds.append(out_ds)

                out_df = out_site.out_per3m_df(self.args, out, y, ytime,y3m)
                df = pd.concat([df, out_df])

                pbar.update(1)  # Increment the progress bar

        # store the results
        if self.args.mlflow:
            mlflow.log_artifact(self.args.ds_save_path)
        with tqdm(desc="Combining datasets", total=len(ds)) as pbar:
            combined_ds = xr.concat(ds, dim="time_since")
            pbar.update(1)
        combined_ds.to_netcdf(self.args.ds_save_path)
        df.to_csv(self.args.result3m_save_path, index=False)
        generate_correlation_csv(self.args,pred_path=self.args.result3m_save_path)

        mem_used_test = psutil.Process().memory_info().rss / 1024 / 1024
        end_time = time.time()
        print(
            "The prediction on the grid area was done and stored at",
            self.args.ds_save_path,
            "with size of",
            out_site.sizeof_fmt(os.path.getsize(self.args.ds_save_path)),
        )
        print(f"Testing time: {end_time - start_time}, Memory used: {mem_used_test} MB")
        print("The job is done", "-" * 30)
