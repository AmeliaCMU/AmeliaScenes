import numpy as np
import pandas as pd

from easydict import EasyDict

# from safeair_scenes.utils import common as C
# from safeair_scenes.utils.trajair import global_masks as G

# from safeair_scenes.scene_scoring.trajair import clustering as cl
# from safeair_scenes.scene_scoring.trajair import cluster_anomaly as ca
# from safeair_scenes.scene_scoring.trajair import individual_features as indf
# from safeair_scenes.scene_scoring.trajair import interaction_features as intf
# from safeair_scenes.scene_scoring.trajair import primitives as P
# from safeair_scenes.scene_scoring.trajair import scores as S


from amelia_scenes.utils import common as C
from safeair_scenes.utils.trajair import global_masks as G

from safeair_scenes.scene_scoring.trajair import clustering as cl
from safeair_scenes.scene_scoring.trajair import cluster_anomaly as ca
from safeair_scenes.scene_scoring.trajair import individual_features as indf
from safeair_scenes.scene_scoring.trajair import interaction_features as intf
from safeair_scenes.scene_scoring.trajair import primitives as P
from safeair_scenes.scene_scoring.trajair import scores as S


def pack_individual_features(
    features: dict,
    blacklist: list = ['num_agents', 'scenario_id'],
    eps: float = 1e-10
) -> pd.DataFrame:
    m = {}
    for k, v in features.items():
        if k in blacklist:
            continue

        if 'waiting_period' in k:
            T = np.asarray([interval for interval, distance, index in v])
            D = np.asarray([distance + eps for interval, distance, index in v])
            # Compute the pace of the agent. The higher the value, the slower it is moving around
            # a conflict point.
            m[k] = np.divide(T, D)
        elif 'acceleration' in k:
            # Note that I separated acceleration values into acceleration and deceleration, since we
            # care about stuff like aborted takeoffs where there's a high acceleration profile
            # followed by an abrupt, high deceleration one.
            m[k] = np.asarray([acc for acc, indices in v])
        elif 'patt' in k:
            # Compute the average distance between an agent's path and the K-closest patterns.
            m[k] = [r for r, h, f in v]
        else:
            # Speed and Jerk should fall in this condition.
            m[k] = np.asarray(v)
    return pd.DataFrame(m)


def compute_individual_scores(
    features: dict,
    agent_types: np.array,
    max_speed: float = 84.0,  # Optimal cruise speed is 230km/h (64m/s), Max speed is 302km/h (84m/s)
    max_acc: float = 30.0,    # Takeoff acceleration 55 knots (~30m/s^2)
    max_jerk: float = 30.0,
    max_in_pattern_dist: float = 5.0,
    max_anomaly_dist: float = 80.0,
    max_score: float = 1000
):
    features_df = pack_individual_features(features)
    N, M = features_df.shape

    # waiting period score
    wp_scores = features_df.waiting_period_L + features_df.waiting_period_C

    # jerk score
    jerk_scores = features_df.jerk.clip(lower=0.0, upper=max_jerk)

    # acceleration score
    acc_rate = features_df.acceleration_rate.clip(lower=0.0, upper=max_acc)
    dec_rate = features_df.deceleration_rate.clip(lower=0.0, upper=max_acc)
    ac_scores = acc_rate + dec_rate + np.sqrt(acc_rate * dec_rate)

    # speed score
    speed_scores = features_df.speed.clip(lower=0.0, upper=max_speed)
    speed_scores += max_speed * features_df.speed_limit

    # in-lane score
    in_pattern_scores = features_df.in_pattern.clip(lower=0.0, upper=max_in_pattern_dist)

    # trajectory anomaly score
    traj_anomaly = features_df.traj_anomaly.clip(lower=0.0, upper=max_anomaly_dist)

    # overall score
    scores = (
        wp_scores
        + ac_scores
        + speed_scores
        + in_pattern_scores
        + jerk_scores
        + traj_anomaly
    ).to_numpy()

    # TODO: move this to interaction score
    # for n, m in zip(range(N), range(N)):
    #     scores[n] += ac_scores[n] * wp_scores[m]
    scores = np.clip(scores, a_min=0.0, a_max=max_score)

    scene_score = scores.max() + scores.mean()
    return scores, scene_score


def compute_interaction_scores(
    features: dict,
    agent_types: np.array,
    max_acc: float = 50.0,    # Takeoff acceleration 55 knots (~30m/s^2)
    max_anomaly_dist: float = 500.0,  # m
    min_time: float = 0.01,
    max_score: float = 1000
):
    def compute_simple_score(idx, eps=1e-5):
        return features.collisions[idx] \
            + min(1.0 / min_time, features.agent_mttcp[idx]) \
            + min(1.0 / min_time, features.scene_mttcp[idx]) \
            + min(1.0 / min_time, features.thw[idx]) \
            + min(1.0 / min_time, features.ttc[idx]) \
            + min(max_acc, features.drac[idx]) \
            + min(max_anomaly_dist, features.traj_pair_anomaly[idx])

    features = EasyDict(features)
    N = features.num_agents
    scores = np.zeros(shape=N)
    for n, (i, j) in enumerate(features.agent_ids):
        scores[i] += compute_simple_score(n)
        scores[j] += compute_simple_score(n)
    scores = np.clip(scores, a_min=0.0, a_max=max_score)

    scene_score = scores.max() + scores.mean()
    return scores, scene_score


def compute_full_score(
    scene: dict,
    individual_features: dict,
    interaction_features: dict,
    min_time: float = 0.01,
    max_period: float = 60.0,
    max_speed: float = 84.0,  # Optimal cruise speed is 230km/h (64m/s), Max speed is 302km/h (84m/s)
    max_acc: float = 50.0,    # Takeoff acceleration 55 knots (~30m/s^2)
    max_jerk: float = 30.0,
    max_in_pattern_dist: float = 5.0,
    max_single_anomaly_dist: float = 300.0,
    max_pair_anomaly_dist: float = 500.0,  # m
    max_score: float = 1000
):
    def compute_simple_individual_score(features):
        # waiting period
        wp_scores = features.waiting_period_L + features.waiting_period_C
        # jerk score
        jerk_scores = features.jerk.clip(lower=0.0, upper=max_jerk)
        # acceleration score
        acc_rate = features.acceleration_rate.clip(lower=0.0, upper=max_acc)
        dec_rate = features.deceleration_rate.clip(lower=0.0, upper=max_acc)
        acc_scores = acc_rate + dec_rate + np.sqrt(acc_rate * dec_rate)
        # speed score
        speed_scores = features.speed.clip(lower=0.0, upper=max_speed)
        speed_scores += max_speed * features.speed_limit
        # in-lane score
        in_pattern_scores = 10.0 * features.in_pattern.clip(lower=0.0, upper=max_in_pattern_dist)
        # trajectory anomaly score
        traj_anomaly = 0.1 * features.traj_anomaly.clip(lower=0.0, upper=max_single_anomaly_dist)
        # overall score
        scores = (
            wp_scores + acc_scores + speed_scores + in_pattern_scores + jerk_scores + traj_anomaly
        ).to_numpy()
        return np.clip(scores, a_min=0.0, a_max=max_score), acc_scores, wp_scores

    ind_features = {}
    for k, v in individual_features.items():
        if k in ['scenario_id']:
            continue
        ind_features[k] = np.asarray(v)
    ind_features = pd.DataFrame(ind_features)
    ind_scores, acc_scores, wp_scores = compute_simple_individual_score(ind_features)

    def compute_simple_interaction_score(features, idx, eps=1e-5):
        return features.collisions[idx] \
            + min(1.0 / min_time, features.agent_mttcp[idx]) \
            + min(1.0 / min_time, features.scene_mttcp[idx]) \
            + min(1.0 / min_time, features.thw[idx]) \
            + min(1.0 / min_time, features.ttc[idx]) \
            + min(max_acc, features.drac[idx]) \
            + 0.1 * min(max_pair_anomaly_dist, features.traj_pair_anomaly[idx])

    int_features = EasyDict(interaction_features)
    N = ind_scores.shape[0]
    int_scores = np.zeros(shape=N)
    for n, (i, j) in enumerate(int_features.agent_ids):
        int_scores[i] += compute_simple_interaction_score(int_features, n) + acc_scores[i] * wp_scores[j]
        int_scores[j] += compute_simple_interaction_score(int_features, n) + acc_scores[j] * wp_scores[i]

    int_scores = np.clip(int_scores, a_min=0.0, a_max=max_score)

    def compute_weights(sequences):
        N = sequences.shape[0]
        weights = np.ones(shape=N)
        for n in range(N):
            mask = np.ones(shape=N).astype(bool)
            mask[n] = False
            agent_traj, others = sequences[n], sequences[mask]
            min_dist = np.linalg.norm(agent_traj[None, ] - others, axis=-1).min()
            weights[n] = 1.0 / (1.0 + min_dist)
        return weights

    sequences = scene['agent_sequences']
    weights = compute_weights(sequences[..., G.XYZ])
    int_scores *= weights
    ind_scores *= weights

    int_scene_score = int_scores.sum()
    ind_scene_score = ind_scores.sum()
    scene_score = ind_scene_score + int_scene_score

    return {
        'scene_score': scene_score,
        'scene_interaction_score': int_scene_score,
        'scene_individual_score': ind_scene_score,
        'interaction_scores': int_scores,
        'individual_scores': ind_scores
    }


def compute_scene_score_online(
    scene, graph_map, graph_cps, aligns, kmeans, max_add=100000, min_timesteps=5, hist_len=20
):
    prims = P.assign_primitives(scene, hist_only=True)
    prims['scene'] = scene

    # Score process (step 2)
    labels = cl.run_labeling_online(prims, aligns, kmeans, max_add, min_timesteps)

    # Score process (step 3)
    anomaly = ca.compute_anomaly_online(prims, labels, min_timesteps)
    single, pair = anomaly['single'], anomaly['pair']

    # Score process (step 4)
    ind_features = C.repack_features(
        indf.compute_individual_features(
            scene, graph_map, graph_cps, hist_len=hist_len, anomaly=single))

    # Score process (step 5)
    int_features = C.repack_features(
        intf.compute_interaction_features(scene, graph_map, graph_cps, pair_anomaly=pair))

    # Score process (step 6)
    scores = S.compute_full_score(scene, ind_features, int_features)

    return scores
