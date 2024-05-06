**Wind Forecasting Repository**

This repository contains scripts and instructions for training and testing a wind forecasting model using DPIRD (Department of Primary Industry and Regional Development) and ECMWF (European Centre for Medium-Range Weather Forecasts) data, along with terrain data.

**Prerequisites**

Before running the scripts, ensure you have the raw datasets ready. The datasets are not provided with this repository due to their large size and proprietary nature. Follow the instructions below to obtain and prepare the required datasets.

**DPIRD Data:**

Download DPIRD data using the Weather API 2.0.
Clean the downloaded data and store it in .csv format for each weather station.
Ensure the file headers include specific columns such as time, airTemperature, relativeHumidity, etc.
Replace missing values with np.nan.
Select the desired features for training the model using the --vars_dpird option in the script.

**ECMWF Data:**

Download ECMWF data (ERA5) using the CDS API.
Use the provided Python script cds.py under ./data_prep_ST/ECMWF/ to download the data.
Specify the desired variables using the --vars_cds option in the script.

**Terrain Data:**

Download terrain data (altitude values) in .tif format.

**Running the Scripts**

The main script is ./scripts/runs3.sh. It supports three modes:

train: Train the model.
test_star: Test the trained model on labelled stations with 10-metre wind profiles.
test_grid: Provide forecasts for the entire area.
If you choose test_grid, specify the testing time using --test_grid_start and --test_grid_end. Monthly testing is recommended for large areas to avoid large output files.

You can adjust parameters such as T_hr (sample duration), L_hr (moving window of y), F_hr (forward window), and S_min (sliding window). Specify model filters using --filters, with recommended values of 4, 8, 16 for default features.

Adjust the batch_size according to your machine specifications.

**Outcome Results**

Some of the outcome results are presented in ./plots_journal for reference. Feel free to explore and analyze these results.
