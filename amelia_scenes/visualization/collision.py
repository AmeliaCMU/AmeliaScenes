
import matplotlib.pyplot as plt
import numpy as np
import itertools

from matplotlib.offsetbox import AnnotationBbox


import amelia_scenes.visualization.common as C
from amelia_scenes.utils import global_masks as G
from amelia_scenes.utils.transform_utils import xy_to_ll
from amelia_scenes.scoring.interactive import compute_collisions

import torch
from geographiclib.geodesic import Geodesic


def plot_scene_collision(
        scene: dict, assets: tuple, predictions: tuple, filename: str = 'tmp.png', dpi=600, reproject=False, projection='EPSG:3857', plot_all=True, coll_threshold=0.2) -> None:
    """ Visualizing marginal predictions for the ego-agent. """

    mm = C.MOTION_COLORS['multi_modal']
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    geodesic = Geodesic.WGS84

    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=1.0, cmap='gray_r')

    hist_len, agent_ids = scene['hist_len'], scene['agent_ids']

    agents_in_scene = scene['agents_in_scene'].detach().cpu().numpy().astype(int)
    agents_interest = np.array(agent_ids)
    other_agents = [i for i in range(scene['num_agents']) if i not in agents_in_scene]
    ego_agent_id = agents_in_scene[scene['ego_agent_ids']]
    ego_agent_ids = scene['ego_agent_ids']
    halo_values = np.array([C.MOTION_COLORS['interest_agent']] * len(agents_interest))
    halo_values[ego_agent_id] = C.MOTION_COLORS['ego_agent']
    if other_agents:
        halo_values[other_agents] = C.MOTION_COLORS['other_agent']
    if not plot_all:
        agents_interest = agents_interest[agents_in_scene]
        halo_values = halo_values[agents_in_scene]
        scene['agent_sequences'] = scene['agent_sequences'][agents_in_scene]
        scene['agent_masks'] = scene['agent_masks'][agents_in_scene]
        scene['agent_types'] = np.array(scene['agent_types'])[agents_in_scene]
        scene['agent_ids'] = np.array(scene['agent_ids'])[agents_in_scene]

    airport_id, scenario_id = scene['airport_id'], scene['scenario_id']

    plt.title(f"{airport_id.upper()} (id {scenario_id})")

    # Plots sequences segmented as 'history', 'future' and 'future entering'
    C.plot_sequences_segmented(
        ax, scene, agents, agents_interest, halo_values=halo_values, reproject=reproject, projection=projection)

    # plot predictions
    pred_scores, mus, sigmas, sequences = predictions
    B, N, T, H, D = mus.shape
    pred_color, pred_lw = C.MOTION_COLORS['pred']
    preds_xy, preds_ll = torch.zeros(size=(N, T, 2)), torch.zeros(size=(N, T, 2))
    for i, agent_id in enumerate(ego_agent_ids):
        scores, mu, sigma = pred_scores[i], mus[i, ..., :2], sigmas[i, ..., :2]

        start_abs = sequences[i, agent_id, hist_len-1, G.XY].detach().cpu().numpy()
        start_heading = sequences[i, agent_id, hist_len-1, G.HD].detach().cpu().numpy()
        preds_xy[agent_id, :hist_len] = sequences[i, agent_id, :hist_len, G.XY]
        preds_ll[agent_id, :hist_len] = sequences[i, agent_id, :hist_len, G.LL]

        best_h = scores[agent_id].argmax().item()
        for h in range(H):
            # Transform predicted trajectories
            pred_ll, pred_xy = xy_to_ll(
                mu[:, :, h], start_abs, start_heading, ref_data, geodesic, return_xyabs=True)
            if h == best_h:
                preds_xy[agent_id, hist_len:] = pred_xy[agent_id, hist_len:]
                preds_ll[agent_id, hist_len:] = pred_ll[agent_id, hist_len:]

            mu_p = mu[:, :, h] + torch.sqrt(sigma[:, :, h])
            mu_n = mu[:, :, h] - torch.sqrt(sigma[:, :, h])
            sigma_p = xy_to_ll(mu_p, start_abs, start_heading, ref_data, geodesic)[agent_id]
            sigma_n = xy_to_ll(mu_n, start_abs, start_heading, ref_data, geodesic)[agent_id]

            score = scores[agent_id, h].item()
            # score = score if not np.isnan(score) else 1.0

            pred = pred_ll[agent_id, hist_len:]
            sigma_p, sigma_n = sigma_p[hist_len:], sigma_n[hist_len:]
            if reproject:
                pred = C.reproject_sequences(pred, projection)
                sigma_p = C.reproject_sequences(sigma_p, projection)
                sigma_n = C.reproject_sequences(sigma_n, projection)
            ax.plot(pred[:, 1], pred[:, 0], color=pred_color, lw=pred_lw, alpha=score)
            ax.fill_between(
                pred[:, 1], sigma_n[:, 0], sigma_p[:, 0], lw=mm[1], alpha=score * 0.50,
                color=mm[0], interpolate=True)

    # plot collision

    agent_collisions = [i for i in range(N) if i not in ego_agent_ids]

    agent_combinations = list(itertools.product(ego_agent_ids, agent_collisions))
    # agent_combinations = list(itertools.combinations(range(N), 2))
    # breakpoint()
    for i, j in agent_combinations:
        if i == j:
            continue
        agent_i = (preds_xy[i], None, None, None, None, None)
        agent_j = (preds_xy[j], None, None, None, None, None)
        coll = compute_collisions(agent_i, agent_j, collision_threshold=.5)  # [hist_len:]

        if np.any(coll == 1.0):
            min_t = np.where(coll == 1.0)[0].min()  # + hist_len
            plt.suptitle(f"Predicted Trajectories Collide from t={min_t}s", color="red")
            # tag += "_coll"
            pred_i, pred_j = preds_ll[i, min_t].unsqueeze(0), preds_ll[j, min_t].unsqueeze(0)
            if reproject:
                pred_i = C.reproject_sequences(pred_i, projection)
                pred_j = C.reproject_sequences(pred_j, projection)

            ax.scatter(pred_i[:, 1], pred_i[:, 0], marker=(10, 1, 0), color='red', s=70, zorder=10)
            ax.scatter(pred_i[:, 1], pred_i[:, 0], marker=(10, 1, 2), color='orange', s=20, zorder=10)

            # ax.scatter(pred_j[:, 1], pred_j[:, 0], marker=(10, 1, 0), color='red', s=70, zorder=10)
            # ax.scatter(pred_j[:, 1], pred_j[:, 0], marker=(10, 1, 2), color='orange', s=20, zorder=10)

    ax.set_xticks([])
    ax.set_yticks([])
    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{filename}', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
