import itertools
import networkx as nx
import numpy as np

import amelia_scenes.utils.common as C
import amelia_scenes.utils.global_masks as G

from easydict import EasyDict
from enum import Enum
from shapely import LineString
from typing import Tuple, Union

METRICS_DEFAULT = {
    'scene_mttcp': float('inf'),
    'agent_mttcp': float('inf'),
    'collisions': 0.0,
}


class Status(Enum):
    UNKNOWN = -1
    OK = 0
    OK_CP = 1
    NOT_OK_STATIONARY = 2
    NOT_OK_DISTANCE = 3
    NOT_OK_SHARED_ZONE = 4
    NOT_OK_HEADING = 5

# --------------------------------------------------------------------------------------------------
# Interaction score wrapper
# --------------------------------------------------------------------------------------------------


def compute_interactive_scores(
    scene: EasyDict,
    hold_lines: np.array,
    norm_constant: float = 60.0
):
    metrics = compute_interactive_metrics(
        scene.agent_sequences,
        scene.agent_types,
        hold_lines[:, 2:4]
    )

    def compute_simple_score(idx, eps=1e-5):
        return metrics.collisions[idx] + \
            min(60, 1.0 / (metrics.agent_mttcp[idx] + eps)) + \
            min(60, 1.0 / (metrics.scene_mttcp[idx] + eps))

    N = scene.num_agents
    scores = np.zeros(shape=N)
    for n, (i, j) in enumerate(metrics.agent_ids):
        agent_types = metrics.agent_types[n]
        scores[i] += C.WEIGHTS[agent_types[0]] * compute_simple_score(n) / norm_constant
        scores[j] += C.WEIGHTS[agent_types[1]] * compute_simple_score(n) / norm_constant

    scene_score = scores.max() + scores.mean()
    return scores, scene_score

# --------------------------------------------------------------------------------------------------
# Interaction metric wrapper
# --------------------------------------------------------------------------------------------------


def compute_interactive_metrics(
    sequences: np.array,
    agent_types: np.array,
    hold_lines: np.array,
    # Returns most critical value from the interactions
    return_critical_value: bool = True,
    # Speed (Km/H) to consider an agent as stationary
    stationary_speed_thresh: float = 10.0,
    # Airport-specific: airplanning.com/post/airport-runways
    agent_to_agent_dist_thresh: float = 1.0,
    # [Arbitrary] 100m distance to a hold-line
    closest_point_dist_thresh: float = 0.1,
    # Separation standards: airservicesaustralia.com
    separation_dist_thresh: float = 0.300,
) -> dict:
    positions = sequences[..., G.XY]
    headings = sequences[..., G.SEQ_IDX['Heading']]
    speeds = sequences[..., G.SEQ_IDX['Speed']]

    N, T, D = positions.shape

    agent_combinations = list(itertools.combinations(range(N), 2))
    status_init = np.asarray([Status.UNKNOWN for _ in agent_combinations])

    metrics = {
        'status': status_init.copy(),
        'agent_types': [(agent_types[i], agent_types[j]) for i, j in agent_combinations],
        'agent_ids': [(i, j) for i, j in agent_combinations],
        'agent_mttcp': [],
        'scene_mttcp': [],
        'collisions': [],
    }

    # Compute the distance from each agent's position to each condlict point.
    # Matrix shape is (num_holdlines, num_agents, timesteps).
    dist_to_conflict_points = C.compute_dists_to_conflict_points(hold_lines, positions)

    for n, (i, j) in enumerate(agent_combinations):
        # Assign a default value to each metric.
        for metric, default_value in METRICS_DEFAULT.items():
            if not metric in metrics:
                continue
            metrics[metric].append(default_value)

        speed_i = speeds[i] * C.KNOTS_TO_KPH
        speed_j = speeds[j] * C.KNOTS_TO_KPH

        # If agents are stationary, assume no interaction is happening so, skip.
        is_stationary_i = speed_i.mean() <= stationary_speed_thresh
        is_stationary_j = speed_j.mean() <= stationary_speed_thresh
        if is_stationary_i and is_stationary_j:
            metrics['status'][n] = Status.NOT_OK_STATIONARY
            continue

        speed_i /= C.HOUR_TO_SECOND  # km/s
        speed_j /= C.HOUR_TO_SECOND

        # If agent are not within a distance threshold from each other, skip. In this case, the
        # distance threshold is the maximum runway extent.
        # NOTE: I'm not sure if we should remove this constraint.
        pos_i, pos_j = positions[i], positions[j]
        D_ij = np.linalg.norm(pos_i - pos_j, axis=1)
        if not np.any(D_ij < agent_to_agent_dist_thresh):
            metrics['status'][n] = Status.NOT_OK_DISTANCE
            continue

        # Check if agents are near a conflict point by some distance threshold.
        dist_cp_i, dist_cp_j = dist_to_conflict_points[:, i], dist_to_conflict_points[:, j]
        in_cp_i = (dist_cp_i < closest_point_dist_thresh).sum(axis=1)
        in_cp_j = (dist_cp_j < closest_point_dist_thresh).sum(axis=1)

        heading_i, heading_j = headings[i], headings[j]

        # Agents state information
        agent_i = (pos_i, speed_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i)
        agent_j = (pos_j, speed_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j)

        # ------------------------------------------------------------------------------------------
        # Compute mTTCP for conflict points in the scene. I divided mTTCP into two metrics:
        #     * Scence mTTCP: Computes the mTTCP of reaching conflict point. For now, the only type
        #                     of conflict point considered are hold lines.
        #
        #     * Agent mTTCP: Also considers if given two agent trajectories pass through the same
        #                    point. Once identified, I calculate the mTTCP from t=0 to t=first time
        #                    one of the agents cross that conflict point. For reference:
        #                    ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9827305
        #
        #     * Collisions: Computes the loss of separation between two agents. Using 0.300m as
        #                   loss of separation which is used for vertical separation. Reference:
        #                   airservicesaustralia.com/about-us/our-services/
        #                       how-air-traffic-control-works/separation-standards/
        # ------------------------------------------------------------------------------------------
        metrics['collisions'][-1] = compute_collisions(
            agent_i=agent_i,
            agent_j=agent_j,
            collision_threshold=separation_dist_thresh,
            return_critical_value=return_critical_value)

        metrics["scene_mttcp"][-1] = compute_scene_mttcp(
            agent_i=agent_i,
            agent_j=agent_j,
            return_critical_value=return_critical_value)[0]

        metrics["agent_mttcp"][-1] = compute_agent_mttcp(
            agent_i=agent_i,
            agent_j=agent_j,
            dist_threshold=separation_dist_thresh,
            return_critical_value=return_critical_value)[0]

        metrics['status'][n] = Status.OK

    return EasyDict(metrics)

# --------------------------------------------------------------------------------------------------
# Specific metrics
# --------------------------------------------------------------------------------------------------


def compute_collisions(
    agent_i: Tuple,
    agent_j: Tuple,
    collision_threshold: float = 0.2,
    return_critical_value: bool = False
) -> Union[np.array, float, int]:
    """ Checks if the agents are either within a threshold position from each other or if their
    trajectory segments intersect. """
    pos_i, vel_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    seg_i = np.stack([pos_i[:-1], pos_i[1:]], axis=1)
    seg_i = [LineString(x) for x in seg_i]

    seg_j = np.stack([pos_j[:-1], pos_j[1:]], axis=1)
    seg_j = [LineString(x) for x in seg_j]

    coll = np.linalg.norm(pos_i - pos_j, axis=-1) <= collision_threshold
    coll = (np.array([False] + [x.intersects(y) for x, y in zip(seg_i, seg_j)])) | coll
    return coll.sum() if return_critical_value else coll


def compute_scene_mttcp(
    agent_i: Tuple,
    agent_j: Tuple,
    return_critical_value: bool = False,
    eps: float = C.EPS
) -> Tuple:
    """ Computes the minimum time to conflict point (mTTCP):

                                   | 𝚫xi(t)     𝚫xj(t)  |
        𝚫TTCP  =       min         |------  ‒‒  ------  |
                  t in {0, tcp}    | 𝚫vi(t)     𝚫vj(t)  |

        between two trajectories that go throgh the same conflict point in a scene at some point
        in their trajectories. Here t=0 is the time the two agents appear in the scene, and t=tcp is
        the first time one of the agents crosses the conflict point. """
    pos_i, vel_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    mttcp, min_t, shared_cps = float('inf'), None, None
    if (len(in_cp_i) == 0 and len(in_cp_j) == 0) or (is_stationary_i and is_stationary_j):
        return mttcp, min_t, shared_cps

    idx_in_cp_i, idx_in_cp_j = np.where(
        in_cp_i > 0)[0], np.where(in_cp_j > 0)[0]
    shared_conflict_points = np.intersect1d(idx_in_cp_i, idx_in_cp_j)
    if shared_conflict_points.shape[0] == 0:
        return mttcp, min_t, shared_cps

    v_i = vel_i + eps
    v_j = vel_j + eps

    # TODO: check if at least one agent is facing the conflict point.
    scene_mttcp = np.inf * np.ones(shared_conflict_points.shape[0])
    min_ts = -1 * np.ones(shared_conflict_points.shape[0]).astype(int)

    # Agent i is already at the conflict point
    if is_stationary_i:
        for n in range(shared_conflict_points.shape[0]):
            scp = shared_conflict_points[n]
            t = dist_cp_j[scp].argmin() + 1
            ttcp_j = dist_cp_j[scp, :t] / v_j[:t]
            scene_mttcp[n] = np.abs(ttcp_j).min()
            min_ts[n] = np.abs(ttcp_j).argmin()

    # Agent j is already at the conlict point
    elif is_stationary_j:
        for n in range(shared_conflict_points.shape[0]):
            scp = shared_conflict_points[n]
            t = dist_cp_i[scp].argmin() + 1
            ttcp_i = dist_cp_i[scp, :t] / v_i[:t]
            scene_mttcp[n] = np.abs(ttcp_i).min()
            min_ts[n] = np.abs(ttcp_i).argmin()

    # Both agents are moving
    else:
        for n in range(shared_conflict_points.shape[0]):
            scp = shared_conflict_points[n]
            t = min(dist_cp_i[scp].argmin(), dist_cp_j[scp].argmin()) + 1
            ttcp_i = dist_cp_i[scp, :t] / v_i[:t]
            ttcp_j = dist_cp_j[scp, :t] / v_j[:t]
            scene_mttcp[n] = np.abs(ttcp_i - ttcp_j).min()
            min_ts[n] = np.abs(ttcp_i - ttcp_j).argmin()

    if return_critical_value:
        ok = np.where(scene_mttcp != np.inf)
        scene_mttcp, min_ts = scene_mttcp[ok], min_ts[ok]
        shared_conflict_points = shared_conflict_points[ok]
        if scene_mttcp.shape[0] == 0:
            return mttcp, min_t, shared_cps
        idx = scene_mttcp.argmin()
        return scene_mttcp.min(), min_ts[idx], shared_conflict_points[idx]
    return scene_mttcp, min_ts, shared_conflict_points


def compute_agent_mttcp(
    agent_i: Tuple,
    agent_j: Tuple,
    dist_threshold: float = 0.5,
    return_critical_value: bool = False,
    eps: float = C.EPS
) -> Tuple:
    """ Computes the minimum time to conflict point (mTTCP):

                                   | 𝚫xi(t)     𝚫xj(t)  |
        𝚫TTCP  =       min         |------  ‒‒  ------  |
                  t in {0, tcp}    | 𝚫vi(t)     𝚫vj(t)  |

    between any two timesteps between two trajectories that are within a distance threshold from each
    other. Here t=0 is the time the two agents appear in the scene, and t=tcp is the first time one
    of the agents crosses the conflict point. """
    pos_i, vel_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    mttcp, min_t, shared_cps = float('inf'), None, None

    # T, 2 -> T, T
    dists = np.linalg.norm(pos_i[:, None, :] - pos_j, axis=-1)
    i_idx, j_idx = np.where(dists <= dist_threshold)

    vals, i_unique = np.unique(i_idx, return_index=True)
    i_idx, j_idx = i_idx[i_unique], j_idx[i_unique]
    if len(i_idx) == 0:
        return mttcp, min_t, shared_cps

    agent_conflict_points = pos_i[i_idx]
    agents_mttcp = np.inf * np.ones(agent_conflict_points.shape[0])
    min_ts = -1 * np.ones(agent_conflict_points.shape[0]).astype(int)

    v_i = vel_i + eps
    v_j = vel_j + eps

    # Agent i is already at the conflict point
    if is_stationary_i:
        t = j_idx[-1] + 1
        ttcp = np.abs(dists[0, :t] / v_j[:t])

    # Agent j is already at the conflict point
    elif is_stationary_j:
        t = i_idx[-1] + 1
        ttcp = np.abs(dists[:t, 0] / v_i[:t])

    # Agents are moving
    else:
        for n, (i, j) in enumerate(zip(i_idx, j_idx)):
            conflict_point = pos_i[i]  # which should be ~pos_j[j]
            t = min(i, j) + 1
            ttcp_i = np.linalg.norm(conflict_point - pos_i[:t], axis=-1) / v_i[:t]
            ttcp_j = np.linalg.norm(conflict_point - pos_j[:t], axis=-1) / v_j[:t]
            ttcp = np.abs(ttcp_i - ttcp_j)
            agents_mttcp[n] = ttcp.min()
            min_ts[n] = ttcp.argmin()

    if return_critical_value:
        ok = np.where(agents_mttcp != np.inf)
        agents_mttcp = agents_mttcp[ok]
        if agents_mttcp.shape[0] == 0:
            return float('inf'), None, None
        idx = agents_mttcp.argmin()
        return agents_mttcp.min(), min_ts[idx], i_idx[idx]
    return agents_mttcp, min_ts, i_idx

# --------------------------------------------------------------------------------------------------
# NOTE: All below is unused for now but DO NOT DELETE.
# --------------------------------------------------------------------------------------------------

def compute_thw(
    agent_i: Tuple,
    agent_j: Tuple,
    leading_agent: np.array,
    dists_ij: np.array,
    mask: np.array,
    return_critical_value: bool = False,
    eps: float = C.EPS
) -> Tuple:
    """ Computes the following measurements:

        Time Headway (THW):
        -------------------
            TWH = t_i - t_j
        where t_i is the time vehicle passes a certain location and t_j is the time the vehicle ahead
        passes that same location.
    """
    pos_i, vel_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    # Mask agent states using valid heading mask
    pos_i, vel_i, heading_i = pos_i[mask], vel_i[mask], heading_i[mask]
    pos_j, vel_j, heading_j = pos_j[mask], vel_j[mask], heading_j[mask]
    leading_agent, dists_ij = leading_agent[mask], dists_ij[mask]

    v_i = vel_i + eps
    v_j = vel_j + eps

    thw = np.inf * np.ones(shape=(pos_i.shape[0]))

    # ...where i is the agent ahead
    i_idx = np.where(leading_agent == 0)[0]
    # ...where j is the agent ahead
    j_idx = np.where(leading_agent == 1)[0]

    t_j = dists_ij / v_j
    t_i = dists_ij / v_i

    if is_stationary_i:
        thw[i_idx] = t_j[i_idx]

    elif is_stationary_j:
        thw[j_idx] = t_i[j_idx]

    else:
        thw[i_idx] = t_j[i_idx]
        thw[j_idx] = t_i[j_idx]

    if return_critical_value:
        ok_idx = np.where((thw != np.inf))[0]
        thw, mask = thw[ok_idx], mask[ok_idx]
        if thw.shape[0] == 0:
            return float('inf'), None
        return thw.min(), mask[thw.argmin()]
    return thw


def compute_ttc(
    agent_i: Tuple,
    agent_j: Tuple,
    leading_agent: np.array,
    dists_ij: np.array,
    mask: np.array,
    separation_len: float = 0.195,
    return_critical_value: bool = False,
    eps: float = C.EPS
) -> Tuple:
    """ Computes the following measurements:

        Time-to-Collision (TTC):
        ------------------------
                  x_j - x_i - l_i
            TTC = ---------------  forall v_i > v_j
                     v_i - v_j

        where x_i, l_i, and v_i are the position, length and speed of the following vehicle and
        x_j and v_j are the position and speed of the leading vehicle.
    """
    pos_i, vel_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    # Mask agent states using valid heading mask
    pos_i, vel_i, heading_i = pos_i[mask], vel_i[mask], heading_i[mask]
    pos_j, vel_j, heading_j = pos_j[mask], vel_j[mask], heading_j[mask]
    leading_agent, dists_ij = leading_agent[mask], dists_ij[mask]

    v_i = vel_i + eps
    v_j = vel_j + eps

    ttc = np.inf * np.ones(shape=(pos_i.shape[0]))

    pos_len_i, pos_len_j = np.zeros_like(pos_i), np.zeros_like(pos_j)
    pos_len_i[:, 0] = pos_i[:, 0] + \
        separation_len * np.cos(np.deg2rad(heading_i))
    pos_len_i[:, 1] = pos_i[:, 1] + \
        separation_len * np.sin(np.deg2rad(heading_i))

    pos_len_j[:, 0] = pos_j[:, 0] + \
        separation_len * np.cos(np.deg2rad(heading_j))
    pos_len_j[:, 1] = pos_j[:, 1] + \
        separation_len * np.sin(np.deg2rad(heading_j))

    # ...where i is the agent ahead
    i_idx = np.where(leading_agent == 0)[0]
    # ...where j is the agent ahead
    j_idx = np.where(leading_agent == 1)[0]
    # ...where i is the leader but j's speed is higher
    v_i_idx = np.intersect1d(i_idx, np.where(v_j > v_i)[0])
    # ...where j is the leader but i's speed is higher
    v_j_idx = np.intersect1d(j_idx, np.where(v_i > v_j)[0])

    # TTC
    dpos_ij = np.linalg.norm(pos_i[v_i_idx] - pos_len_j[v_i_idx], axis=-1)
    dpos_ji = np.linalg.norm(pos_j[v_j_idx] - pos_len_i[v_j_idx], axis=-1)

    if is_stationary_i:
        ttc[v_i_idx] = dpos_ij / v_j[v_i_idx]

    elif is_stationary_j:
        ttc[v_j_idx] = dpos_ji / v_i[v_j_idx]

    else:
        ttc[v_i_idx] = dpos_ij / (v_j[v_i_idx] - v_i[v_i_idx])
        ttc[v_j_idx] = dpos_ji / (v_i[v_j_idx] - v_j[v_j_idx])

    if return_critical_value:
        ok_idx = np.where((ttc != np.inf))[0]
        ttc, mask = ttc[ok_idx], mask[ok_idx]
        if ttc.shape[0] == 0:
            return float('inf'), None
        return ttc.min(), mask[ttc.argmin()]
    return ttc


def compute_drac(
    agent_i: Tuple,
    agent_j: Tuple,
    leading_agent: np.array,
    dists_ij: np.array,
    mask: np.array,
    return_critical_value: bool = False,
    eps: float = C.EPS
):
    """ Computes the following measurements:

        Deceleration Rate to Avoid a Crash (DRAC):
        -----------------------------------------
                    (v_j - v_i) ** 2
            DRAC = ------------------
                      2 (x_i - x_j)
        the average delay of a road user to avoid an accident at given velocities and distance
        between vehicles, where i is the leader and j is the follower.
    """
    pos_i, vel_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    # Mask agent states using valid heading mask
    pos_i, vel_i, heading_i = pos_i[mask], vel_i[mask], heading_i[mask]
    pos_j, vel_j, heading_j = pos_j[mask], vel_j[mask], heading_j[mask]
    leading_agent, dists_ij = leading_agent[mask], dists_ij[mask]

    v_i = vel_i + eps
    v_j = vel_j + eps

    drac = np.zeros(shape=(pos_i.shape[0]))

    # ...where i is the agent ahead
    i_idx = np.where(leading_agent == 0)[0]
    # ...where j is the agent ahead
    j_idx = np.where(leading_agent == 1)[0]
    # ...where i is the leader but j's speed is higher
    v_i_idx = np.intersect1d(i_idx, np.where(v_j > v_i)[0])
    # ...where j is the leader but i's speed is higher
    v_j_idx = np.intersect1d(j_idx, np.where(v_i > v_j)[0])

    dpos_ij = np.linalg.norm(pos_i[v_i_idx] - pos_j[v_i_idx], axis=-1) + eps
    drac[v_i_idx] = ((v_j[v_i_idx] - v_i[v_i_idx]) ** 2) / (2 * dpos_ij)

    dpos_ji = np.linalg.norm(pos_j[v_j_idx] - pos_i[v_j_idx], axis=-1) + eps
    drac[v_j_idx] = ((v_i[v_j_idx] - v_j[v_j_idx]) ** 2) / (2 * dpos_ji)

    if return_critical_value:
        ok_idx = np.where((drac != np.inf))[0]
        drac, mask = drac[ok_idx], mask[ok_idx]
        if drac.shape[0] == 0:
            return 0.0, None
        return drac.max(), mask[drac.argmax()]
    return drac

# TODO: all below need to be verified


def compute_shared_zone_interacion_state(
    agent_i, agent_j, leading_agent, dists_ij, mask, separation_len=0.195, return_critical_value=False,
    eps=0.0001
):
    """ Computes the following measurements:

        Time Headway (THW):
        -------------------
            TWH = t_i - t_j
        where t_i is the time vehicle passes a certain location and t_j is the time the vehicle ahead
        passes that same location.

        Time-to-Collision (TTC):
        ------------------------
                  x_j - x_i - l_i
            TTC = ---------------  forall v_i > v_j
                     v_i - v_j

        where x_i, l_i, and v_i are the position, length and speed of the following vehicle and
        x_j and v_j are the position and speed of the leading vehicle.

        Deceleration Rate to Avoid a Crash (DRAC):
        -----------------------------------------
                    (v_j - v_i) ** 2
            DRAC = ------------------
                      2 (x_i - x_j)
        the average delay of a road user to avoid an accident at given velocities and distance
        between vehicles, where i is the leader and j is the follower.
    """
    pos_i, vel_i, heading_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    # Mask agent states using valid heading mask
    pos_i, vel_i, heading_i = pos_i[mask], vel_i[mask], heading_i[mask]
    pos_j, vel_j, heading_j = pos_j[mask], vel_j[mask], heading_j[mask]
    leading_agent, dists_ij = leading_agent[mask], dists_ij[mask]

    v_i = vel_i + eps
    v_j = vel_j + eps

    thw = np.inf * np.ones(shape=(pos_i.shape[0]))
    ttc = np.inf * np.ones(shape=(pos_i.shape[0]))
    drac = np.zeros(shape=(pos_i.shape[0]))

    pos_len_i, pos_len_j = np.zeros_like(pos_i), np.zeros_like(pos_j)
    pos_len_i[:, 0] = pos_i[:, 0] + separation_len * np.cos(np.deg2rad(heading_i))
    pos_len_i[:, 1] = pos_i[:, 1] + separation_len * np.sin(np.deg2rad(heading_i))

    pos_len_j[:, 0] = pos_j[:, 0] + separation_len * np.cos(np.deg2rad(heading_j))
    pos_len_j[:, 1] = pos_j[:, 1] + separation_len * np.sin(np.deg2rad(heading_j))

    # ...where i is the agent ahead
    i_idx = np.where(leading_agent == 0)[0]
    # ...where j is the agent ahead
    j_idx = np.where(leading_agent == 1)[0]
    # ...where i is the leader but j's speed is higher
    v_i_idx = np.intersect1d(i_idx, np.where(v_j > v_i)[0])
    # ...where j is the leader but i's speed is higher
    v_j_idx = np.intersect1d(j_idx, np.where(v_i > v_j)[0])

    # THW
    t_j = dists_ij / v_j
    t_i = dists_ij / v_i

    # TTC
    dpos_ij = np.linalg.norm(pos_i[v_i_idx] - pos_len_j[v_i_idx], axis=-1)
    dpos_ji = np.linalg.norm(pos_j[v_j_idx] - pos_len_i[v_j_idx], axis=-1)

    if is_stationary_i:
        thw[i_idx] = t_j[i_idx]

        ttc[v_i_idx] = dpos_ij / v_j[v_i_idx]

    elif is_stationary_j:
        thw[j_idx] = t_i[j_idx]

        ttc[v_j_idx] = dpos_ji / v_i[v_j_idx]

    else:
        thw[i_idx] = t_j[i_idx]
        thw[j_idx] = t_i[j_idx]

        ttc[v_i_idx] = dpos_ij / (v_j[v_i_idx] - v_i[v_i_idx])
        ttc[v_j_idx] = dpos_ji / (v_i[v_j_idx] - v_j[v_j_idx])

    dpos_ij = np.linalg.norm(pos_i[v_i_idx] - pos_j[v_i_idx], axis=-1)
    drac[v_i_idx] = ((v_j[v_i_idx] - v_i[v_i_idx]) ** 2) / (2 * dpos_ij)

    dpos_ji = np.linalg.norm(pos_j[v_j_idx] - pos_i[v_j_idx], axis=-1)
    drac[v_j_idx] = ((v_i[v_j_idx] - v_j[v_j_idx]) ** 2) / (2 * dpos_ji)

    # Return most relevant interacion
    if return_critical_value:
        ok = np.where((thw != np.inf))[0]
        thw = thw[ok]
        if thw.shape[0] == 0:
            thw, thw_idx = float('inf'), None
        else:
            thw, thw_idx = thw.min(), mask[ok][thw.argmin()]

        ok = np.where((ttc != np.inf))[0]
        ttc = ttc[ok]
        if ttc.shape[0] == 0:
            ttc, ttc_idx = float('inf'), None
        else:
            ttc, ttc_idx = ttc.min(), mask[ok][ttc.argmin()]

        ok = np.where((drac != np.inf))[0]
        drac = drac[ok]
        if drac.shape[0] == 0:
            drac, drac_idx = 0.0, None
        else:
            drac, drac_idx = drac.max(), mask[ok][drac.argmax()]

        return (thw, thw_idx), (ttc, ttc_idx), (drac, drac_idx)
    return thw, ttc, drac
