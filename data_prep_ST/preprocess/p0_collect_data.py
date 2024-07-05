import pandas as pd
import os
import numpy as np
from data_provider.data_collection import (
    SiteMeta,
    SiteData,
    NearSites,
    time_session,
    dataSiteSession,
    norm_time,
)

def download_data(stationCode, stationName, start,end,api_key):
    variables = [
        "time",
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
        "wind_3m_speed",
        "wind_3m_degN",
        "wind_10m_speed",
        "wind_10m_degN",
    ]
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    data = pd.DataFrame(columns=variables)
    # get the start and end time for each session
    timeSessions = time_session(start, end, '15 m')
    for timeSession in timeSessions:
        # get the wind data for each session
        windRawData_site_session = SiteData(
            timeSession, stationCode, '15 m', api_key
        )
        windData_site_session = dataSiteSession(
            stationName, variables, windRawData_site_session
        )
        data = pd.concat([data, windData_site_session], axis=0)
        print("finished processing", timeSession[0].strftime("%Y-%m-%d %H:%M"))
    data.replace("nan", np.nan, inplace=True)
    data.drop_duplicates()
    data = data.dropna(how="all").reset_index(drop=True)

    return data

def update_dpird_dataset(start,end,dataset_path='/mnt/science1/fchen/dataset_DPIRD',
                         dpird_site_list='./data_prep_ST/make_grid/all_station_coordinates.csv'):

    api_key = 'oWLaFKFB8FZwwpNHihe5wXUo2Kggtxag'
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    print(f'updating dpird sites data from {start} to {end}, the sites are listed under {dpird_site_list}')

    dpird_sites_df = pd.read_csv(dpird_site_list)
    for index,row in dpird_sites_df.iterrows():
        site_code = row['code']
        site_name = row['name']
        site_meta = SiteMeta(site_code, api_key)
        if site_meta.get("collection") is None:
            print(f'{site_name} has no data')
            continue
        file_path = os.path.join(dataset_path,f'{site_name}.csv')
        if os.path.isfile(file_path):
            print(site_name,'data is available')
            data = pd.read_csv(file_path)
            data['time'] = pd.to_datetime(data['time'])
            next_time = data["time"].max() + pd.to_timedelta("15 m")
            if data["time"].max() < end:
                print("but it is not the latest one, updating")
                data_append = download_data(site_code, site_name, next_time,end,api_key)
                data = (
                    pd.concat([data, data_append])
                    .dropna(how="all")
                    .reset_index(drop=True)
                )
                data.to_csv(file_path,
                            index=False,
                            )
        else:
            print(site_name, "data downloading from scratch")

            data_site = download_data(site_code, site_name, start,end,api_key)
            data_site.to_csv(
                file_path, index=False
            )