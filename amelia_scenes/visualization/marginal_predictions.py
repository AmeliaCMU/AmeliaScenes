import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import AnnotationBbox
from typing import Tuple
from matplotlib import cm

import amelia_scenes.visualization.common as C
from amelia_scenes.utils import global_masks as G

import torch
from geographiclib.geodesic import Geodesic
from src.utils.transform_utils import xy_to_ll


def plot_scene_marginal(
        scene: dict, assets: tuple, predictions: tuple, filename: str = 'tmp.png', dpi=600, reproject=False, projection='EPSG:3857', plot_all=True) -> None:
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

    agents_in_scene = scene['agents_in_scene'].detach().cpu().numpy().astype(int).tolist()
    agents_interest = np.array(agent_ids)
    other_agents = [i for i in range(scene['num_agents']) if i not in agents_in_scene]
    ego_agent_ids = scene['ego_agent_ids']
    halo_values = np.array([C.MOTION_COLORS['interest_agent']] * len(agents_interest))
    halo_values[ego_agent_ids] = C.MOTION_COLORS['ego_agent']
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
                preds_ll[agent_id, hist_len:] = pred_ll[agent_id, hist_len:]
                preds_xy[agent_id, hist_len:] = pred_xy[agent_id, hist_len:]

            mu_p = mu[:, :, h] + torch.sqrt(sigma[:, :, h])
            mu_n = mu[:, :, h] - torch.sqrt(sigma[:, :, h])
            sigma_p = xy_to_ll(mu_p, start_abs, start_heading, ref_data, geodesic)[agent_id]
            sigma_n = xy_to_ll(mu_n, start_abs, start_heading, ref_data, geodesic)[agent_id]

            score = scores[agent_id, h].item()
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

    ax.set_xticks([])
    ax.set_yticks([])
    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{filename}', dpi=dpi,
                bbox_inches='tight', pad_inches=0)
    plt.close()


# def plot_scene_marginal_fast(
#         gt_history: np.array, gt_future: np.array, pred_trajectories: np.array, pred_scores: np.array,
#         sigmas: np.array, maps: Tuple, ll_extent: Tuple, agent_types: np.array = None, tag: str = 'temp.png',
#         ego_id: int = 0, out_dir: str = './out', agents_interest: list = [], change_projection: bool = False,
#         projection: str = 'EPSG:3857') -> None:
#     """ Tool for visualizing marginal model predictions for the ego-agent. """
#     mm = C.MOTION_COLORS['multi_modal']

#     north, east, south, west = ll_extent
#     if change_projection:
#         new_projection = C.transform_extent(ll_extent, C.MAP_CRS, projection)
#         north, east, south, west = new_projection
#     fig, movement_plot = plt.subplots()

#     bkg, agent_maps = maps[0], maps[1]
#     # Display global map
#     movement_plot.imshow(bkg, zorder=0, extent=[
#                          west, east, south, north], alpha=1.0, cmap='gray_r')

#     N, T, H, D = pred_trajectories.shape

#     # Iterate through agent in the scene
#     all_lon, all_lat, scores = [], [], []
#     sigmasn, sigmasp = sigmas[..., 0], sigmas[..., 1]
#     zipped = zip(gt_history, gt_future, pred_scores, pred_trajectories, sigmasn, sigmasp, agent_types)
#     for n, (gt_hist, gt_fut, hscores, preds, sigmas_p, sigmas_n, agent_type) in enumerate(zipped):

#         # Get heading at last point of trajectory history.
#         gt_hist_ll, gt_heading = gt_hist[:, 1:], gt_hist[-1, 0]
#         if change_projection:
#             gt_hist_ll = C.reproject_sequences(gt_hist_ll, projection)
#             gt_fut = C.reproject_sequences(gt_fut, projection)

#         lon, lat = gt_hist_ll[-1, 1], gt_hist_ll[-1, 0]
#         if lon == 0 or lat == 0:
#             continue

#         agent_type = int(agent_type)
#         # Place plane on last point of ground truth sequence
#         icon = agent_maps[agent_type]
#         img = C.plot_agent(icon, gt_heading, zoom=C.ZOOM[agent_type])
#         if n in agents_interest:
#             if n != ego_id:
#                 movement_plot.scatter(
#                     lon, lat, color='#F29C3A',
#                     alpha=C.MOTION_COLORS['ego_agent'][1], s=160)

#         ab = AnnotationBbox(img, (lon, lat), frameon=False)
#         movement_plot.add_artist(ab)
#         # Plot past sequence
#         movement_plot.plot(
#             gt_hist_ll[:, 1], gt_hist_ll[:,
#                                          0], color=C.MOTION_COLORS['gt_hist'][0],
#             lw=C.MOTION_COLORS['gt_hist'][1])

#         movement_plot.plot(
#             gt_fut[:, 1], gt_fut[:, 0], color=C.MOTION_COLORS['gt_future'][0],
#             lw=C.MOTION_COLORS['gt_future'][1], linestyle='dashed')

#         if n == ego_id:
#             movement_plot.scatter(
#                 lon, lat, color=C.MOTION_COLORS['ego_agent'][0],
#                 alpha=C.MOTION_COLORS['ego_agent'][1], s=300)

#             # Iterate through prediction heads
#             for h in range(H):
#                 score = hscores[h]  # Get confidence score for head
#                 scores.append(score)
#                 if score > 0.02:
#                     pred = preds[:, h]  # Get best sampled predicted trajectory
#                     s_p = sigmas_p[:, h, :]  # Positive sigma
#                     s_n = sigmas_n[:, h, :]  # Negative sigma

#                     if change_projection:
#                         pred = C.reproject_sequences(pred, projection)
#                         s_p = C.reproject_sequences(s_p, projection)
#                         s_n = C.reproject_sequences(s_n, projection)

#                     # Plot predicted trajectory.
#                     movement_plot.plot(
#                         pred[:, 1], pred[:, 0], color=C.MOTION_COLORS['pred'][0],
#                         lw=C.MOTION_COLORS['pred'][1], alpha=score.item())

#                     # Plot motion distribution
#                     movement_plot.fill_between(
#                         pred[:, 1], s_n[:, 0], s_p[:,
#                                                    0], lw=mm[1], alpha=score.item() * 0.65,
#                         color=mm[0], interpolate=True)

#             all_lon.append(lon)
#             all_lat.append(lat)

#     # Plot movement
#     movement_plot.set_xticks([])
#     movement_plot.set_yticks([])

#     # if lat < abs(north) and lat > south and lon > west and lon < east:
#     #     movement_plot.axis(
#     #         [np.min(all_lon) - C.OFFSET, np.max(all_lon) + C.OFFSET,
#     #          np.min(all_lat) - C.OFFSET, np.max(all_lat) + C.OFFSET])

#     # Set figure bbox around the predicted trajectory
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.savefig(f'{out_dir}/{tag}.png', dpi=800,
#                 bbox_inches='tight', pad_inches=0)
#     plt.close()


# def plot_scene_marginal(
#     gt_history: np.array, gt_future: np.array, pred_trajectories: np.array, pred_scores: np.array,
#     sigmas: np.array, maps: Tuple, ll_extent: Tuple, tag: str = 'temp.png', ego_id: int = 0,
#     out_dir: str = './out'
# ) -> None:
#     """ Tool for visualizing motion predictions using samples.  """
#     mm = C.MOTION_COLORS['multi_modal']
#     # Format color tuples
#     mm = (mm[0][0], mm[0][1], mm[0][2], mm[1])
#     fade = colors.to_rgb(mm) + (0.0,)
#     pred_fade = colors.LinearSegmentedColormap.from_list('my', [fade, mm])
#     mm = (mm, 0.9)
#     C.MOTION_COLORS['alphas'] = np.array(
#         [0.15*((30-t)/20) for t in range(10, 30)])

#     north, east, south, west = ll_extent
#     fig, movement_plot = plt.subplots()

#     bkg, agent_maps = maps[0], maps[1]
#     # Display global map
#     movement_plot.imshow(bkg, zorder=0, extent=[
#                          west, east, south, north], alpha=1.0, cmap='gray_r')

#     N, T, H, D = pred_trajectories.shape

#     # Iterate through agent in the scene
#     all_lon, all_lat, scores = [], [], []
#     sigmasn, sigmasp = sigmas[..., 0], sigmas[..., 1]

#     zipped = zip(gt_history, gt_future, pred_scores,
#                  pred_trajectories, sigmasn, sigmasp)

#     for n, (gt_hist, gt_fut, hscores, preds, sigmas_p, sigmas_n) in enumerate(zipped):

#         # Get heading at last point of trajectory history.
#         gt_hist_ll, gt_heading = gt_hist[:, 1:], gt_hist[-1, 0]
#         lon, lat = gt_hist_ll[-1, 1], gt_hist_ll[-1, 0]
#         if lon == 0 or lat == 0:
#             continue

#         # Place plane on last point of ground truth sequence
#            # Place plane on last point of ground truth sequence
#         img = C.plot_agent(agent_maps[C.AIRCRAFT],
#                            gt_heading, zoom=C.ZOOM[C.AIRCRAFT])
#         ab = AnnotationBbox(img, (lon, lat), frameon=False)
#         movement_plot.add_artist(ab)

#         # Plot past sequence
#         movement_plot.plot(
#             gt_hist_ll[:, 1], gt_hist_ll[:,
#                                          0], color=C.MOTION_COLORS['gt_hist'][0],
#             lw=C.MOTION_COLORS['gt_hist'][1])

#         movement_plot.plot(
#             gt_fut[:, 1], gt_fut[:, 0], color=C.MOTION_COLORS['gt_future'][0],
#             lw=C.MOTION_COLORS['gt_future'][1], linestyle='dashed')

#         if n == ego_id:
#             movement_plot.scatter(
#                 lon, lat, color=C.MOTION_COLORS['ego_agent'][0],
#                 alpha=C.MOTION_COLORS['ego_agent'][1], s=300)

#             # Iterate through prediction heads
#             for h in range(H):
#                 score = hscores[h]  # Get confidence score for head
#                 scores.append(score)
#                 if score > 0.02:
#                     pred = preds[:, h]  # Get best sampled predicted trajectory
#                     s_p = sigmas_p[:, h, :]  # Positive sigma
#                     s_n = sigmas_n[:, h, :]  # Negative sigma
#                     s = ((s_p - s_n)/2)  # Sigma
#                     # Plot predicted trajectory.
#                     movement_plot.plot(
#                         pred[:, 1], pred[:, 0], color=C.MOTION_COLORS['pred'][0],
#                         lw=C.MOTION_COLORS['pred'][1], alpha=score.item())

#                     # Add other samples based on the confidence of the prediction (with a max of 10)
#                     samples = int(score.item() * 500)
#                     for _ in range(samples):
#                         # Get normal distribution
#                         f = abs(np.random.normal(0, 0.3))

#                         lon_sp, lat_sp = (
#                             pred[:, 1] + f * s[:, 1], pred[:, 0] + f * s[:, 0])
#                         C.plot_faded_prediction(
#                             lon_sp, lat_sp, movement_plot, gradient=C.MOTION_COLORS['alphas'],
#                             color=pred_fade)

#                         lon_sn, lat_sn = (
#                             pred[:, 1] - f * s[:, 1], pred[:, 0] - f * s[:, 0])
#                         C.plot_faded_prediction(
#                             lon_sn, lat_sn, movement_plot, gradient=C.MOTION_COLORS['alphas'],
#                             color=pred_fade)

#             all_lon.append(lon)
#             all_lat.append(lat)

#     # Plot movement
#     movement_plot.set_xticks([])
#     movement_plot.set_yticks([])

#     if lat < abs(north) and lat > south and lon > west and lon < east:
#         movement_plot.axis(
#             [np.min(all_lon) - C.OFFSET, np.max(all_lon) + C.OFFSET,
#              np.min(all_lat) - C.OFFSET, np.max(all_lat) + C.OFFSET])

#     # Set figure bbox around the predicted trajectory
#     plt.tight_layout()
#     plt.savefig(f'{out_dir}/{tag}.png', dpi=800)
#     plt.close()


# def plot_scene_marginal_with_hist(
#     gt_history: np.array, pred_trajectories: np.array, pred_scores: np.array, sigmas: np.array,
#     maps: Tuple, ll_extent: Tuple, gt_future: np.array = None, tag: str = 'temp.png', ego_id: int = 0,
#     out_dir: str = './out'
# ) -> None:
#     """ Function for visualizing the prediction of a scene. """
#     mm = C.MOTION_COLORS['multi_modal']
#     fade = colors.to_rgb(mm) + (0.0,)
#     pred_fade = colors.LinearSegmentedColormap.from_list('my', [fade, mm])
#     mm = (mm, 0.9)
#     C.MOTION_COLORS['alphas'] = np.array(
#         [0.15*((30-t)/20) for t in range(10, 30)])

#     north, east, south, west = ll_extent

#     # Save states
#     fig, (movement_plot, speed_plot) = plt.subplots(
#         2, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [3, 1]})

#     bkg_ground, ac_map = maps[0], maps[1]
#     # Display global map
#     movement_plot.imshow(
#         bkg_ground, zorder=0, extent=[west, east, south, north], alpha=1.0, cmap='gray_r')

#     N, T, H, D = pred_trajectories.shape
#     # Iterate through agent in the scene
#     all_lon, all_lat, vel_sigma, vel_mu, scores = [], [], [], [], []

#     zipped = zip(gt_history, gt_future, pred_scores,
#                  pred_trajectories, sigmas[..., 0], sigmas[..., 1])
#     for n, (gt_traj, pred_gt, hscores, preds, sigmas_p, sigmas_n) in enumerate(zipped):
#         # Get heading at last point of trajectory.
#         gt, heading = gt_traj[:, 1:], gt_traj[-1, 0]
#         lon, lat = gt[-1, 1], gt[-1, 0]
#         if lon == 0 or lat == 0:
#             continue

#         # Place plane on last point of ground truth sequence
#         img = C.plot_agent(ac_map, heading)
#         ab = AnnotationBbox(img, (lon, lat), frameon=False)
#         movement_plot.add_artist(ab)

#         # Plot past sequence
#         movement_plot.plot(
#             gt[:, 1], gt[:, 0], color=C.MOTION_COLORS['gt_hist'][0], lw=C.MOTION_COLORS['gt_hist'][1])
#         movement_plot.plot(
#             pred_gt[:, 1], pred_gt[:, 0], color=C.MOTION_COLORS['gt_future'][0],
#             lw=C.MOTION_COLORS['gt_future'][1], linestyle='dashed')

#         if n == ego_id:
#             movement_plot.scatter(
#                 lon, lat, color=C.MOTION_COLORS['ego_agent'][0], alpha=C.MOTION_COLORS['ego_agent'][1],
#                 s=300)

#             history_velocity = C.get_velocity(gt)
#             future_velocity = C.get_velocity(pred_gt)

#             # Iterate through prediction header
#             for h in range(H):
#                 score = hscores[h]  # Get confidence score for head
#                 scores.append(score)

#                 if score > 0.02:
#                     pred = preds[:, h]  # Get best sampled predicted trajectory
#                     s_p = sigmas_p[:, h, :]  # Positive sigma
#                     s_n = sigmas_n[:, h, :]  # Negative sigma
#                     s = ((s_p - s_n) / 2)  # Sigma

#                     predicted_velocity = C.get_velocity(pred[:])
#                     vel_mu.append(predicted_velocity)

#                     # Plot predicted trajectory.
#                     movement_plot.plot(
#                         pred[:, 1], pred[:, 0], color=C.MOTION_COLORS['pred'][0],
#                         lw=C.MOTION_COLORS['pred'][1], alpha=score.item())

#                     # Evaluate positive boundary for prediction distribution
#                     positive_sigma = pred + s
#                     vel_sp = C.get_velocity(positive_sigma)
#                     vel_sigma.append(vel_sp)

#                     # Evaluate negative boundary for prediction distribution
#                     negative_sigma = pred - s
#                     vel_sn = C.get_velocity(negative_sigma)
#                     vel_sigma.append(vel_sn)

#                     # Plot motion distribution
#                     movement_plot.fill_between(
#                         pred[:, 1], s_n[:, 0], s_p[:, 0], lw=mm[1], alpha=score.item() * 0.65,
#                         color=mm[0], interpolate=True)

#             all_lon.append(lon)
#             all_lat.append(lat)

#     # Plot movement
#     movement_plot.set_xticks([])
#     movement_plot.set_yticks([])

#     if lat < abs(north) and lat > south and lon > west and lon < east:
#         movement_plot.axis(
#             [np.min(all_lon) - C.OFFSET, np.max(all_lon) + C.OFFSET,
#              np.min(all_lat) - C.OFFSET, np.max(all_lat) + C.OFFSET])

#     C.plot_speed_histogram(speed_plot, vel_mu, vel_sigma,
#                            history_velocity, future_velocity)

#     # Set figure bbox around the predicted trajectory
#     plt.savefig(f'{out_dir}/{tag}.png', dpi=500, bbox_inches='tight')
#     plt.close()
