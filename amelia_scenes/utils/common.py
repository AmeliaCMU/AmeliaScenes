import cv2
import os
import numpy as np
import pandas as pd
import pickle
import random

from typing import Tuple

np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# TODO: same as the global_masks.py, avoid copying these global variables to each module.
AIRCRAFT = 0
VEHICLE = 1
UNKNOWN = 2

EPS = 1e-5

WEIGHTS = {
    AIRCRAFT: 1.0,
    VEHICLE: 0.2,
    UNKNOWN: 0.4
}

KNOTS_TO_MPS = 0.51444445
KNOTS_TO_KPH = 1.852
HOUR_TO_SECOND = 3600
KMH_TO_MS = 1/3.6

SUPPORTED_AIRPORTS = [
    "kbos",
    "kdca",
    "kewr",
    "kjfk",
    "klax",
    "kmdw",
    "kmsy",
    "ksea",
    "ksfo",
    "panc",
    "katl",
    "kpit",
    "kdfw",
    "ksan",
    "kcle",
    "kmke"
]

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))


def get_sorted_order(values):
    return np.argsort(values)[::-1]


def get_random_order(num_agents, agent_valid, seed):
    # NOTE: an overkill way to create a randomized list of valid agent indeces + interpolated
    # agent indeces. This is so that the __getitem__ function can choose a random ego-agent.
    agents_in_scene = np.asarray(list(range(num_agents)))
    valid_agents = agents_in_scene[agent_valid]
    # random.seed(seed)
    random.shuffle(valid_agents)

    invalid_agents = agents_in_scene[~agent_valid]
    # random.seed(seed)
    random.shuffle(invalid_agents)
    random_agents_in_scene = np.asarray(valid_agents.tolist() + invalid_agents.tolist())
    return random_agents_in_scene

# TODO: debug this function!


def impute(seq: pd.DataFrame, seq_len: int, imputed_flag: float = 1.0) -> pd.DataFrame:
    """ Imputes missing data via linear interpolation.

    Inputs
    ------
        seq[pd.DataFrame]: trajectory sequence to be imputed.
        seq_len[int]: length of the trajectory sequence.

    Output
    ------
        seq[pd.DataFrame]: trajectory sequence after imputation.
    """
    # Create a list from starting frame to ending frame in agent sequence
    conseq_frames = set(range(int(seq[0, 0]), int(seq[-1, 0])+1))
    # Create a list of the actual frames in the agent sequence. There may be missing data from which
    # we need to interpolate.
    actual_frames = set(seq[:, 0])
    # Compute the difference between the lists. The difference represents the missing data points.
    missing_frames = list(sorted(conseq_frames - actual_frames))
    # Insert nan rows where the missing data is. Then, interpolate.
    if len(missing_frames) > 0:
        seq = pd.DataFrame(seq)
        agent_id = seq.loc[0, 1]
        agent_type = seq.loc[0, 9]
        for missing_frame in missing_frames:
            df1 = seq[:missing_frame]
            df2 = seq[missing_frame:]
            df1.loc[missing_frame] = [
                missing_frame, agent_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                agent_type, imputed_flag, np.nan, np.nan]
            # df1.loc[missing_frame] = [
            #     missing_frame, agent_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            #     agent_type, np.nan, np.nan]
            seq = pd.concat([df1, df2]).astype(float)
        seq = seq.interpolate(method='linear').to_numpy()[:seq_len]
    return seq


def compute_dists_to_conflict_points(conflict_points, positions):
    dists = np.linalg.norm(conflict_points[:, None, None, :] - positions, axis=-1)
    return dists


def get_available_airports(in_data_dir: str) -> list:
    available_airports = []
    for airport in SUPPORTED_AIRPORTS:
        if os.path.isdir(os.path.join(in_data_dir, airport)):
            available_airports.append(airport)
    return available_airports
