import itertools
import numpy as np
import pandas as pd

import amelia_scenes.utils.common as C
import amelia_scenes.utils.global_masks as G

from easydict import EasyDict
from typing import Tuple, Union

from amelia_scenes.utils.common import WEIGHTS, EPS, KM_TO_M

# --------------------------------------------------------------------------------------------------
# Kinematic score wrapper
# --------------------------------------------------------------------------------------------------


def pack_kinematic_features(
    metrics: dict,
    keys_to_pack: list = [
        'waiting_period_L', 'waiting_period_C', 'acceleration_rate', 'deceleration_rate', 'speed', 
        'jerk'],
    eps: float = EPS
):
    m = {}
    for k, v in metrics.items():
        if k not in keys_to_pack:
            continue
        if 'waiting_period' in k:
            T = np.asarray([t for t, d, i in v])
            D = np.asarray([d + eps for t, d, i in v]) #* scale
            m[k] = np.divide(T, D)
        elif 'celeration' in k:
            m[k] = np.asarray([a for a, i in v])
        else:
            m[k] = np.asarray(v)
    return pd.DataFrame(m)


def get_weights(
    idx_arr: np.array
) -> np.array:
    return np.array([WEIGHTS[idx] for idx in idx_arr])


def compute_kinematic_scores(
    scene: dict,
    hold_lines: np.array,
    features: dict = {},
    max_wp: float = 20.0,    # Roughly, 50 seconds waiting 5m from the hold line (x2)
    max_speed: float = 100.0, # Takeoff speed is 100-160 knots, so setting max to 90 m/s (160 knots) 
    max_acc: float = 70.0,   # Approx, 3.4 m/s^2 is the max acceleration of a commercial aircraft
    max_jerk: float = 10.0,  # Set empirically, not sure about this value
    acc_scale: float = 0.1,  # Ugh, somewhat arbitrary scaling factor for acceleration 
    speed_scale: float = 0.1, # Ugh, somewhat arbitrary scaling factor for speed
):
    # Check if features have been computed
    if not features:
        features = compute_kinematic_features(scene, hold_lines)

    features_df = pack_kinematic_features(features)
    N, M = features_df.shape

    idxs = features['agent_idxs']
    weights = get_weights(scene['agent_types'])[idxs]

    # waiting period score
    wp_scores = features_df.iloc[:, :2].sum(axis=1).clip(lower=0.0, upper=2 * max_wp) / (2 * max_wp)
    wp_scores = features_df.waiting_period_L + features_df.waiting_period_C
    wp_scores = wp_scores.clip(lower=0.0, upper=max_wp) #/ max_wp

    # jerk score
    jerk_scores = features_df.jerk.clip(lower=0.0, upper=max_jerk) #/ max_jerk

    # acceleration score TODO: verify this feature
    acc_scores = features_df.acceleration_rate.clip(lower=0.0, upper=max_acc) 
    dec_scores = features_df.deceleration_rate.clip(lower=0.0, upper=max_acc) 
    ac_scores = acc_scale * (acc_scores + dec_scores + acc_scores * dec_scores)

    # speed score NOTE: not a very informative feature
    speed_scores = speed_scale * features_df.speed.clip(lower=0.0, upper=max_speed)

    # overall score
    scores = weights * (wp_scores + ac_scores + speed_scores + jerk_scores).to_numpy()
    
    # NOTE: Debugging agent pairing score. Scale is too big
    scene_score = scores.max() + scores.mean()
    
    # TODO: figure out hot to handle this better later. 
    num_agents = scene['num_agents']
    if scores.shape[0] != num_agents:
        scores_with_invalid_agents = np.zeros(num_agents)
        scores_with_invalid_agents[idxs] = scores
        scores = scores_with_invalid_agents
    return scores, scene_score

# --------------------------------------------------------------------------------------------------
# Kinematic metric wrapper
# --------------------------------------------------------------------------------------------------

def compute_kinematic_features(scene: dict, hold_lines: np.array) -> dict:
    sequences, masks, valids = scene['agent_sequences'], scene['agent_masks'], scene['agent_valid']
    positions, speeds = sequences[..., G.XY], sequences[..., G.SEQ_IDX['Speed']]

    metrics = {
        'waiting_period_L': [],
        'waiting_period_C': [],
        'acceleration_rate': [],
        'deceleration_rate': [],
        'speed': [],
        'jerk': [],
        'agent_idxs': []
    }

    N, T, D = positions.shape
    for n, (pos, speed, mask, valid) in enumerate(zip(positions, speeds, masks, valids)):
        if not valid:    
            continue

        t = np.arange(pos.shape[0])[mask]
        dt = (t[1:] - t[:-1]) + EPS # ensure no division by zero
        
        pos, speed = pos[mask], speed[mask]
        if pos.shape[0] < 2 or speed.shape[0] < 2:
            continue

        metrics['agent_idxs'].append(n)
        
        speed_max, speed = compute_speed(speed) # Speed is in M/S
        metrics['speed'].append(speed_max)

        ar, dr, acc = compute_acceleration_profile(speed, dt)
        metrics['acceleration_rate'].append(ar)
        metrics['deceleration_rate'].append(dr)
        
        jerk_max, jerk = compute_jerk(acc, dt)
        metrics['jerk'].append(jerk_max)

        wp_L, wp_C = compute_waiting_period(pos, hold_lines)
        metrics['waiting_period_L'].append(wp_L)
        metrics['waiting_period_C'].append(wp_C)

    return metrics

# --------------------------------------------------------------------------------------------------
# Specific metrics
# --------------------------------------------------------------------------------------------------


def compute_waiting_period(
    sequence: np.array,
    conflict_points: np.array,
    motion_thresh: float = 0.01  # KM
) -> Tuple:
    """ Computes the longest time interval an agent is stationary at a conflict point. """
    # int_L, dist_L: longest interval and corresopnding distance
    # int_C, dist_C: interval with closest distance to a conflict point and corresponding distance
    wp_int_L, wp_dist_L, wp_idx_L = np.zeros(shape=1), np.inf * np.ones(shape=1), None
    wp_int_C, wp_dist_C, wp_idx_C = np.zeros( shape=1), np.inf * np.ones(shape=1), None

    dp = np.zeros(shape=sequence.shape[0])
    dp[1:] = np.linalg.norm(sequence[1:] - sequence[:-1], axis=-1)
    is_waiting = dp <= motion_thresh
    if sum(is_waiting) > 0:
        is_waiting = np.hstack([[False], is_waiting, [False]])
        is_waiting = np.diff(is_waiting.astype(int))

        starts = np.where(is_waiting == 1)[0]
        ends = np.where(is_waiting == -1)[0]
        se_idxs = [(s, e) for s, e in zip(starts, ends)]

        intervals = np.array([end - start for start, end in zip(starts, ends)])
        dists_cps = np.linalg.norm(conflict_points[:, None] - sequence[starts], axis=-1).min(axis=0)

        idx = intervals.argmax()
        wp_int_L, wp_dist_L, wp_idx_L = intervals[idx], dists_cps[idx], se_idxs[idx]

        idx = dists_cps.argmin()
        wp_int_C, wp_dist_C, wp_idx_C = intervals[idx], dists_cps[idx], se_idxs[idx]

    return (wp_int_L, wp_dist_L * KM_TO_M, wp_idx_L), (wp_int_C, wp_dist_C * KM_TO_M, wp_idx_C)


def compute_jerk(acceleration: np.array, dt: float = 1.0) -> Union[np.array, np.array]:
    """ Computes the jerk from the acceleration profile and time delta. """
    # Here, using np.gradient instead of np.diff to keep the same shape
    da = np.gradient(acceleration, axis=0)
    assert da.shape == dt.shape, "Acceleration and dt must have the same shape."
    jerk = da / dt
    return jerk.max(), jerk


def compute_acceleration_profile(speed: np.array, dt: np.array) -> Union[Tuple, np.array]:
    """ Computes the acceleration profile from the speed (m/s) and time delta. """
    def get_acc_sums(acc: np.array, idx: np.array) -> Tuple[np.array, np.array]:
        diff = idx[1:] - idx[:-1]
        diff = np.array([-1] + np.where(diff > 1)[0].tolist() + [diff.shape[0]])
        se_idxs = [(idx[s+1], idx[e]+1) for s, e in zip(diff[:-1], diff[1:])]
        sums = np.array([acc[s:e].sum() for s, e in se_idxs])
        return sums, se_idxs

    max_acc, max_dec = 0.0, 0.0
    idx_acc, idx_dec = None, None
    
    dv = np.diff(speed, axis=0)
    # dv = np.gradient(speed, axis=0)
    assert dv.shape == dt.shape, "Speed and dt must have the same shape."
    acc = dv / dt  # M/S^2

    # If the agent is accelerating or maintaining acceleration
    dr_idx = np.where(acc < 0.0)[0]
    if dr_idx.shape[0] == 0:
        max_acc = acc.max()
        idx_acc = (0, 59)
    # If the agent is decelerating
    elif dr_idx.shape[0] == acc.shape[0]:
        max_dec = abs(acc.min())
        idx_dec = (0, 59)
    # If both
    else:
        max_dec, idx_dec = get_acc_sums(acc, dr_idx)
        max_dec = abs(max_dec.min())
        idx_dec = idx_dec[max_dec.argmin()]

        ar_idx = np.where(acc >= 0.0)[0]
        max_acc, idx_acc = get_acc_sums(acc, ar_idx)
        max_acc = max_acc.max()
        idx_acc = idx_acc[max_acc.argmax()]

    return (max_acc, idx_acc), (max_dec, idx_dec), acc


def compute_speed(speed: np.array) -> Union[np.array, np.array]:
    """ Since speed is given, this wrapper only converts it from knots to m/s. """
    speed = speed * C.KNOTS_TO_MPS
    return speed.max(), speed
