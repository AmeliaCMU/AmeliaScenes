
import numpy as np
import pandas as pd

import amelia_scenes.utils.common as C
import amelia_scenes.utils.global_masks as G

from easydict import EasyDict
from typing import Tuple, Union

from amelia_scenes.utils.common import WEIGHTS

# --------------------------------------------------------------------------------------------------
# Kinematic score wrapper
# --------------------------------------------------------------------------------------------------


def pack_kinematic_metrics(
    metrics: dict,
    keys_to_pack: list = [
        'waiting_period_L', 'waiting_period_C', 'acceleration_rate', 'deceleration_rate', 'speed', 
        'jerk', 'traj_anomaly'],
    scale: float = 1000.0,
    eps: float = 1e-10
):
    m = {}
    for k, v in metrics.items():
        if not k in keys_to_pack:
            continue
        if 'waiting_period' in k:
            T = np.asarray([t for t, d, i in v])
            D = np.asarray([d + eps for t, d, i in v]) * scale
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
    scene: EasyDict,
    hold_lines: np.array,
    max_wp: float = 60.0,
    max_speed: float = 100.0,
    max_jerk: float = 10.0,
    max_acc: float = 80.0,
    max_score: float = 1000
):
    metrics = compute_kinematic_metrics(scene.agent_sequences, hold_lines[:, 2:4])
    metrics_df = pack_kinematic_metrics(metrics)
    N, M = metrics_df.shape

    weights = get_weights(scene.agent_types)

    # waiting period score
    wp_scores = metrics_df.iloc[:, :2].sum(axis=1).clip(lower=0.0, upper=2 * max_wp) / (2 * max_wp)

    # acceleration score
    acc_scores = metrics_df.acceleration_rate.clip(lower=0.0, upper=max_acc) / max_acc
    dec_scores = metrics_df.deceleration_rate.clip(lower=0.0, upper=max_acc) / max_acc
    ac_scores = acc_scores + dec_scores + acc_scores * dec_scores

    # speed score
    speed_scores = metrics_df.speed.clip(lower=0.0, upper=max_speed) / max_speed

    # jerk score
    jerk_scores = metrics_df.jerk.clip(lower=0.0, upper=max_jerk) / max_jerk

    # overall score
    scores = (wp_scores + ac_scores + speed_scores + jerk_scores).to_numpy()
    for n, m in zip(range(N), range(N)):
        scores[n] += ac_scores[n] * wp_scores[m]
    scores *= weights
    scores = np.clip(scores, a_min=0.0, a_max=max_score)

    scene_score = scores.max() + scores.mean()
    return scores, scene_score

# --------------------------------------------------------------------------------------------------
# Kinematic metric wrapper
# --------------------------------------------------------------------------------------------------


def compute_kinematic_metrics(sequences: np.array, hold_lines: np.array) -> dict:
    positions = sequences[..., G.XY]
    speeds = sequences[..., G.SEQ_IDX['Speed']]

    metrics = {
        'waiting_period_L': [],
        'waiting_period_C': [],
        'acceleration_rate': [],
        'deceleration_rate': [],
        'speed': [],
        'jerk': [],
    }

    N, T, D = positions.shape
    for n in range(N):
        pos, speed = positions[n], speeds[n]

        speed_max, speed = compute_speed(speed)
        metrics['speed'].append(speed_max)

        wp_L, wp_C = compute_waiting_period(pos, hold_lines)
        metrics['waiting_period_L'].append(wp_L)
        metrics['waiting_period_C'].append(wp_C)

        ar, dr, acc = compute_acceleration_profile(speed)
        metrics['acceleration_rate'].append(ar)
        metrics['deceleration_rate'].append(dr)

        jerk_max, jerk = compute_jerk(acc)
        metrics['jerk'].append(jerk_max)

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

    return (wp_int_L, wp_dist_L, wp_idx_L), (wp_int_C, wp_dist_C, wp_idx_C)


def compute_jerk(
    acceleration: np.array,
    dt: float = 1.0
) -> Union[np.array, np.array]:
    jerk = np.gradient(acceleration, dt)
    return jerk.max(), jerk


def compute_acceleration_profile(
    speed: np.array,  # assumed to be in KM/S
    dt: float = 1.0
) -> Union[Tuple, np.array]:
    def get_acc_sums(acc: np.array, idx: np.array) -> Tuple[np.array, np.array]:
        diff = idx[1:] - idx[:-1]
        diff = np.array([-1] + np.where(diff > 1)[0].tolist() + [diff.shape[0]])
        se_idxs = [(idx[s+1], idx[e]+1) for s, e in zip(diff[:-1], diff[1:])]
        sums = np.array([acc[s:e].sum() for s, e in se_idxs])
        return sums, se_idxs

    max_acc, max_dec = 0.0, 0.0
    idx_acc, idx_dec = None, None

    acc = (speed[1:] - speed[:-1]) / dt  # KM/S^2
    dr_idx = np.where(acc < 0.0)[0]

    # If the agent is accelerating or maintaining acceleration
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


def compute_speed(
    speed: np.array
) -> Union[np.array, np.array]:
    speed = speed * C.KNOTS_TO_MPS
    return speed.max(), speed
