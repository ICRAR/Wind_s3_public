#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --output=/dir_of_output_file.log

. /home/fchen/workenv_a400/bin/activate

cd /home/fchen/Wind/ || exit # replace with your main directory
echo 'hello'
export CUDA_AVAILABLE_DEVICES=0

START=("2022-1-1" "2022-4-1" "2022-7-1" "2022-10-1" "2023-1-1" "2023-4-1" "2023-7-1" "2023-10-1")
END=("2022-4-1" "2022-7-1" "2022-10-1" "2023-1-1" "2023-4-1" "2023-7-1" "2023-10-1" "2024-1-1")

for mode in train test_star; do
  python -u run_S3.py \
        --flag "$mode" \
        --customise_grid \
        --customise_grid_config -35.4 -32.0 115.0 118.4 \
        --gpu_ix 1 \
        --vars_terrain \
        --vars_dpird wind_3m_u wind_3m_v airTemperature relativeHumidity \
        --vars_cds u10 v10 msl \
        --labels wind_10m_u wind_10m_v \
        --T_hr 48 \
        --L_hr 4 \
        --F_hr 4 \
        --S_min 15 \
        --filters 4 8 16 \
        --epochs 200 \
        --batch_size 64 \
        --terrain_src_path '/raw_terrain_DEM_data.tif' \
        --cds_src_path '/raw_downloaded_ECMWF_dataset.nc' \
        --dpird_src_path '/folder_path_to_DPIRD_dataset' \
        --st_data_path '/dest_folder_to_save_formatted_data' \
        --result_path '/dest_folder_to_save_results' \
        --model_path '/dest_folder_to_save_model_statedict'
done

for ((i=0; i<${#START[@]}; i++)); do
    start=${START[$i]}
    end=${END[$i]}

    python -u run_S3.py \
          --customise_grid \
          --customise_grid_config -35.4 -32.0 115.0 118.4 \
          --flag test_grid \
          --gpu_ix 1 \
          --vars_terrain \
          --vars_dpird wind_3m_u wind_3m_v airTemperature relativeHumidity \
          --vars_cds u10 v10 msl \
          --labels wind_10m_u wind_10m_v \
          --test_grid_start $start \
          --test_grid_end $end \
          --T_hr 48 \
          --L_hr 4 \
          --F_hr 4 \
          --S_min 15 \
          --filters 4 8 16 \
          --epochs 200 \
          --batch_size 64 \
          --terrain_src_path '/raw_terrain_DEM_data.tif' \
          --cds_src_path '/raw_downloaded_ECMWF_dataset.nc' \
          --dpird_src_path '/folder_path_to_DPIRD_dataset' \
          --st_data_path '/dest_folder_to_save_formatted_data' \
          --result_path '/dest_folder_to_save_results' \
          --model_path '/dest_folder_to_save_model_statedict'
done
