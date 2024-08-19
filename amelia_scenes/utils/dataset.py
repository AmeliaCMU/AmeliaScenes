import cv2
import imageio
import glob
import json
import numpy as np
import os
import pickle
import random


from datetime import datetime, timezone
from easydict import EasyDict
from math import floor
from natsort import natsorted

from typing import Tuple

from amelia_scenes.utils import common


def load_assets(input_dir: str, airport: str) -> Tuple:
    # Graph
    graph_data_dir = os.path.join(input_dir, "graph_data_a10v01os", airport)
    print(f"Loading graph data from: {graph_data_dir}")
    pickle_map_filepath = os.path.join(graph_data_dir, "semantic_graph.pkl")
    with open(pickle_map_filepath, 'rb') as f:
        graph_pickle = pickle.load(f)
        hold_lines = graph_pickle['hold_lines']
        graph_nx = graph_pickle['graph_networkx']
        # pickle_map = temp_dict['map_infos']['all_polylines'][:]

    assets_dir = os.path.join(input_dir, "assets")
    print(f"Loading assets from: {assets_dir}")

    # Map asset
    raster_map_filepath = os.path.join(assets_dir, airport, "bkg_map.png")
    raster_map = cv2.imread(raster_map_filepath)
    raster_map = cv2.resize(raster_map, (raster_map.shape[0]//2, raster_map.shape[1]//2))
    raster_map = cv2.cvtColor(raster_map, cv2.COLOR_BGR2RGB)

    # Reference file
    limits_filepath = os.path.join(assets_dir, airport, 'limits.json')
    with open(limits_filepath, 'r') as fp:
        ref_data = EasyDict(json.load(fp))
    alt = ref_data.limits.Altitude
    espg = ref_data.espg_4326
    limits = (espg.north, espg.east, espg.south, espg.west, alt.min, alt.max)
    ref_data = [ref_data.ref_lat, ref_data.ref_lon, ref_data.range_scale]

    # Agent assets
    agents = {
        common.AIRCRAFT: imageio.imread(os.path.join(assets_dir, "ac.png")),
        common.VEHICLE: imageio.imread(os.path.join(assets_dir, "vc.png")),
        common.UNKNOWN: imageio.imread(os.path.join(assets_dir, "uk_ac.png"))
    }
    return raster_map, hold_lines, graph_nx, (limits, ref_data), agents


def load_data(airport: str, split: str, version: str = 'a10v08'):
    """ Loads processed Amelia scenarios and visualization assets. """
    input_dir = os.path.join(common.BASE_DIR, 'amelia')
    assert os.path.exists(input_dir), f"Path not found: {input_dir}"
    traj_data_dir = os.path.join(input_dir, f'traj_data_{version}')
    assert os.path.exists(traj_data_dir), f"Path not found: {traj_data_dir}"
    proc_data_dir = os.path.join(traj_data_dir, "proc_trajectories")
    assert os.path.exists(proc_data_dir), f"Path not found: {proc_data_dir}"
    data_filepath = os.path.join(traj_data_dir, "splits", f"{split}_splits", f"{airport}_day.txt")
    with open(data_filepath, "r") as f:
        subdirs = f.read().splitlines()
    assert len(subdirs) > 0, f"No files in {proc_data_dir}"

    scenario_subdirs = [os.path.join(proc_data_dir, d) for d in subdirs]

    print(f"Processing airport: {airport.upper()}")
    # Get list of scenarios to process
    scenarios = []
    for scenario_subdir in scenario_subdirs:
        scenarios_list = glob.glob(f"{scenario_subdir.removesuffix('.csv')}/*.pkl", recursive=True)
        scenarios += scenarios_list

    assets = load_assets(input_dir=input_dir, airport=airport)
    return scenarios, assets


def load_blacklist(data_prep: EasyDict, airport_list: list):
    """ Goes through the blacklist files and gets all blacklisted filenames for each airport. If no
        blacklists have been created, it'll only return an empty dictionary.

    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    # TODO: add blacklist path to configs/paths; this should handle automatic dir creation (I think)
    blacklist_dir = os.path.join(data_prep.in_data_dir, 'blacklist')
    os.makedirs(blacklist_dir, exist_ok=True)
    blacklist = {}
    for airport in airport_list:
        blacklist_file = os.path.join(blacklist_dir, f"{airport}_{data_prep.split_type}.txt")
        blacklist[airport] = []
        if os.path.exists(blacklist_file):
            with open(blacklist_file, 'r') as f:
                blacklist[airport] = f.read().splitlines()
    return blacklist


def flatten_blacklist(blacklist: dict):
    blacklist_list = []
    for k, v in blacklist.items():
        blacklist_list += v
    return blacklist_list


def remove_blacklisted(blacklist: list, file_list: list):
    for duplicate in list(set(blacklist) & set(file_list)):
        file_list.remove(duplicate)
    return file_list


def get_airport_files(airport: str, data_prep: dict):
    """ Gets the airport data file list from the specified input directory and returns a random set.

    Inputs
    ------
        airport[str]: airport IATA
        data_prep[dict]: dictionary containing data preparation parameters.
    """

    in_data_dir = os.path.join(data_prep.in_data_dir, airport)
    airport_files = [os.path.join(airport, fp) for fp in os.listdir(in_data_dir)]
    airport_files = natsorted(airport_files)

    random.seed(data_prep.seed)
    random.shuffle(airport_files)
    return airport_files


def create_random_splits(data_prep: EasyDict, airport_list: list):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all
    `train_airports` and create a train-val-test split for each, by keeping floor(75%) of the files
    into the train-val and the remaining floor(25%) into the test set. Files are randomly selected.

    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    n_train, n_val, n_test = data_prep.random_splits.train_val_test
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_prep.out_data_dir, f"{split}_splits")
        os.makedirs(split_dir, exist_ok=True)

    for airport in airport_list:
        airport_files = get_airport_files(airport, data_prep)

        N_train = floor(len(airport_files) * n_train)
        N_val = floor(len(airport_files) * n_val)

        filename = f"{airport}_{data_prep.split_type}"

        # Write the out the splits
        train_list = airport_files[:N_train]
        with open(f"{data_prep.out_data_dir}/train_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(train_list))

        val_list = airport_files[N_train:N_train+N_val]
        with open(f"{data_prep.out_data_dir}/val_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(val_list))

        test_list = airport_files[N_train+N_val:]
        with open(f"{data_prep.out_data_dir}/test_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(test_list))


def create_day_splits(data_prep: dict, airport_list: list):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all
    `seen_airports` and create a train-val-test split for each, by keeping floor(75%) of the days
    into the train-val and the remaining floor(25%) into the test set.

    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    n_train, n_val, n_test = data_prep.day_splits.train_val_test
    train_val_perc = data_prep.day_splits.train_val_perc
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_prep.out_data_dir, f"{split}_splits")
        os.makedirs(split_dir, exist_ok=True)

    for airport in airport_list:
        # Collect all airport files in current airport and get the unique days for which data was
        # collected.
        airport_files = np.asarray(get_airport_files(airport, data_prep))
        days_per_file = np.asarray([datetime.fromtimestamp(
            int(f.split('/')[-1].split('.')[0].split('_')[-1]),
            tz=timezone.utc).day for f in airport_files])

        days = np.unique(days_per_file)
        num_days = days.shape[0]

        np.random.seed(data_prep.seed)

        # Make sure test set does not contain days "seen" during training.
        train_val_days = np.random.choice(days, size=int(train_val_perc * num_days), replace=False)
        test_days = list(set(days.tolist()).symmetric_difference(train_val_days.tolist()))

        train_val_idx = np.in1d(days_per_file, train_val_days)
        train_val_files = airport_files[train_val_idx].tolist()

        filename = f"{airport}_{data_prep.split_type}"

        N_train = floor(len(train_val_files) * n_train)
        train_list = train_val_files[:N_train]
        with open(f"{data_prep.out_data_dir}/train_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(train_list))

        val_list = train_val_files[N_train:]
        with open(f"{data_prep.out_data_dir}/val_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(val_list))

        test_idx = np.in1d(days_per_file, test_days)
        test_list = airport_files[test_idx].tolist()
        with open(f"{data_prep.out_data_dir}/test_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(test_list))


def create_month_splits(data_prep: dict, airport_list: list):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all
    `seen_airports` and create a train-val-test split for each, by keeping floor(75%) of the days
    into the train-val and the remaining floor(25%) into the test set.

    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    n_train, n_val, n_test = data_prep.month_splits.train_val_test
    train_val_perc = data_prep.month_splits.train_val_perc
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_prep.out_data_dir, f"{split}_splits")
        os.makedirs(split_dir, exist_ok=True)

    for airport in airport_list:
        # Collect all airport files in current airport and get the unique days for which data was
        # collected.
        airport_files = np.asarray(get_airport_files(airport, data_prep))
        month_per_file = np.asarray([datetime.fromtimestamp(
            int(f.split('/')[-1].split('.')[0].split('_')[-1]),
            tz=timezone.utc).month for f in airport_files])

        months = np.unique(month_per_file)
        num_months = months.shape[0]

        np.random.seed(data_prep.seed)

        # Make sure test set does not contain months "seen" during training.
        train_val_months = np.random.choice(
            months, size=int(train_val_perc * num_months), replace=False)
        test_months = list(set(months.tolist()).symmetric_difference(train_val_months.tolist()))

        train_val_idx = np.in1d(month_per_file, train_val_months)
        train_val_files = airport_files[train_val_idx].tolist()

        filename = f"{airport}_{data_prep.split_type}"

        N_train = floor(len(train_val_files) * n_train)
        train_list = train_val_files[:N_train]
        with open(f"{data_prep.out_data_dir}/train_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(train_list))

        val_list = train_val_files[N_train:]
        with open(f"{data_prep.out_data_dir}/val_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(val_list))

        test_idx = np.in1d(month_per_file, test_months)
        test_list = airport_files[test_idx].tolist()
        with open(f"{data_prep.out_data_dir}/test_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(test_list))
