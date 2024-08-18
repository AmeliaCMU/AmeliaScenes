import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import AnnotationBbox
from typing import Tuple

import amelia_scenes.visualization.common as C


def plot_scene_marginal_fast(
        gt_history: np.array, gt_future: np.array, pred_trajectories: np.array, pred_scores: np.array,
        sigmas: np.array, maps: Tuple, ll_extent: Tuple, agent_types: np.array = None, tag: str = 'temp.png',
        ego_id: int = 0, out_dir: str = './out', agents_interest: list = [], change_projection: bool = False, 
        projection: str = 'EPSG:3857') -> None:
    """ Tool for visualizing marginal model predictions for the ego-agent. """
    mm = C.MOTION_COLORS['multi_modal']

    north, east, south, west = ll_extent
    if change_projection:
        new_projection = C.transform_extent(ll_extent, C.MAP_CRS, projection)
        north, east, south, west = new_projection
    fig, movement_plot = plt.subplots()

    bkg, agent_maps = maps[0], maps[1]
    # Display global map
    movement_plot.imshow(bkg, zorder=0, extent=[
                         west, east, south, north], alpha=1.0, cmap='gray_r')

    N, T, H, D = pred_trajectories.shape

    # Iterate through agent in the scene
    all_lon, all_lat, scores = [], [], []
    sigmasn, sigmasp = sigmas[..., 0], sigmas[..., 1]

    zipped = zip(gt_history, gt_future, pred_scores,
                 pred_trajectories, sigmasn, sigmasp, agent_types)
    for n, (gt_hist, gt_fut, hscores, preds, sigmas_p, sigmas_n, agent_type) in enumerate(zipped):

        # Get heading at last point of trajectory history.
        gt_hist_ll, gt_heading = gt_hist[:, 1:], gt_hist[-1, 0]
        if change_projection:
            gt_hist_ll = C.reproject_sequences(gt_hist_ll, projection)
            gt_fut = C.reproject_sequences(gt_fut, projection)

        lon, lat = gt_hist_ll[-1, 1], gt_hist_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type = int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agent_maps[agent_type]
        img = C.plot_agent(icon, gt_heading, zoom=C.ZOOM[agent_type])
        if n in agents_interest:
            if n != ego_id:
                movement_plot.scatter(
                    lon, lat, color='#F29C3A',
                    alpha=C.MOTION_COLORS['ego_agent'][1], s=160)

        ab = AnnotationBbox(img, (lon, lat), frameon=False)
        movement_plot.add_artist(ab)
        # Plot past sequence
        movement_plot.plot(
            gt_hist_ll[:, 1], gt_hist_ll[:,
                                         0], color=C.MOTION_COLORS['gt_hist'][0],
            lw=C.MOTION_COLORS['gt_hist'][1])

        movement_plot.plot(
            gt_fut[:, 1], gt_fut[:, 0], color=C.MOTION_COLORS['gt_future'][0],
            lw=C.MOTION_COLORS['gt_future'][1], linestyle='dashed')

        if n == ego_id:
            movement_plot.scatter(
                lon, lat, color=C.MOTION_COLORS['ego_agent'][0],
                alpha=C.MOTION_COLORS['ego_agent'][1], s=300)

            # Iterate through prediction heads
            for h in range(H):
                score = hscores[h]  # Get confidence score for head
                scores.append(score)
                if score > 0.02:
                    pred = preds[:, h]  # Get best sampled predicted trajectory
                    s_p = sigmas_p[:, h, :]  # Positive sigma
                    s_n = sigmas_n[:, h, :]  # Negative sigma

                    if change_projection:
                        pred = C.reproject_sequences(pred, projection)
                        s_p = C.reproject_sequences(s_p, projection)
                        s_n = C.reproject_sequences(s_n, projection)

                    # Plot predicted trajectory.
                    movement_plot.plot(
                        pred[:, 1], pred[:, 0], color=C.MOTION_COLORS['pred'][0],
                        lw=C.MOTION_COLORS['pred'][1], alpha=score.item())

                    # Plot motion distribution
                    movement_plot.fill_between(
                        pred[:, 1], s_n[:, 0], s_p[:,
                                                   0], lw=mm[1], alpha=score.item() * 0.65,
                        color=mm[0], interpolate=True)

            all_lon.append(lon)
            all_lat.append(lat)

    # Plot movement
    movement_plot.set_xticks([])
    movement_plot.set_yticks([])

    # if lat < abs(north) and lat > south and lon > west and lon < east:
    #     movement_plot.axis(
    #         [np.min(all_lon) - C.OFFSET, np.max(all_lon) + C.OFFSET,
    #          np.min(all_lat) - C.OFFSET, np.max(all_lat) + C.OFFSET])

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{out_dir}/{tag}.png', dpi=800,
                bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_scene_marginal(
    gt_history: np.array, gt_future: np.array, pred_trajectories: np.array, pred_scores: np.array,
    sigmas: np.array, maps: Tuple, ll_extent: Tuple, tag: str = 'temp.png', ego_id: int = 0,
    out_dir: str = './out'
) -> None:
    """ Tool for visualizing motion predictions using samples.  """
    mm = C.MOTION_COLORS['multi_modal']
    # Format color tuples
    mm = (mm[0][0], mm[0][1], mm[0][2], mm[1])
    fade = colors.to_rgb(mm) + (0.0,)
    pred_fade = colors.LinearSegmentedColormap.from_list('my', [fade, mm])
    mm = (mm, 0.9)
    C.MOTION_COLORS['alphas'] = np.array(
        [0.15*((30-t)/20) for t in range(10, 30)])

    north, east, south, west = ll_extent
    fig, movement_plot = plt.subplots()

    bkg, agent_maps = maps[0], maps[1]
    # Display global map
    movement_plot.imshow(bkg, zorder=0, extent=[
                         west, east, south, north], alpha=1.0, cmap='gray_r')

    N, T, H, D = pred_trajectories.shape

    # Iterate through agent in the scene
    all_lon, all_lat, scores = [], [], []
    sigmasn, sigmasp = sigmas[..., 0], sigmas[..., 1]

    zipped = zip(gt_history, gt_future, pred_scores,
                 pred_trajectories, sigmasn, sigmasp)
    for n, (gt_hist, gt_fut, hscores, preds, sigmas_p, sigmas_n) in enumerate(zipped):

        # Get heading at last point of trajectory history.
        gt_hist_ll, gt_heading = gt_hist[:, 1:], gt_hist[-1, 0]
        lon, lat = gt_hist_ll[-1, 1], gt_hist_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        # Place plane on last point of ground truth sequence
           # Place plane on last point of ground truth sequence
        img = C.plot_agent(agent_maps[C.AIRCRAFT],
                           gt_heading, zoom=C.ZOOM[C.AIRCRAFT])
        ab = AnnotationBbox(img, (lon, lat), frameon=False)
        movement_plot.add_artist(ab)

        # Plot past sequence
        movement_plot.plot(
            gt_hist_ll[:, 1], gt_hist_ll[:,
                                         0], color=C.MOTION_COLORS['gt_hist'][0],
            lw=C.MOTION_COLORS['gt_hist'][1])

        movement_plot.plot(
            gt_fut[:, 1], gt_fut[:, 0], color=C.MOTION_COLORS['gt_future'][0],
            lw=C.MOTION_COLORS['gt_future'][1], linestyle='dashed')

        if n == ego_id:
            movement_plot.scatter(
                lon, lat, color=C.MOTION_COLORS['ego_agent'][0],
                alpha=C.MOTION_COLORS['ego_agent'][1], s=300)

            # Iterate through prediction heads
            for h in range(H):
                score = hscores[h]  # Get confidence score for head
                scores.append(score)
                if score > 0.02:
                    pred = preds[:, h]  # Get best sampled predicted trajectory
                    s_p = sigmas_p[:, h, :]  # Positive sigma
                    s_n = sigmas_n[:, h, :]  # Negative sigma
                    s = ((s_p - s_n)/2)  # Sigma
                    # Plot predicted trajectory.
                    movement_plot.plot(
                        pred[:, 1], pred[:, 0], color=C.MOTION_COLORS['pred'][0],
                        lw=C.MOTION_COLORS['pred'][1], alpha=score.item())

                    # Add other samples based on the confidence of the prediction (with a max of 10)
                    samples = int(score.item() * 500)
                    for _ in range(samples):
                        # Get normal distribution
                        f = abs(np.random.normal(0, 0.3))

                        lon_sp, lat_sp = (
                            pred[:, 1] + f * s[:, 1], pred[:, 0] + f * s[:, 0])
                        C.plot_faded_prediction(
                            lon_sp, lat_sp, movement_plot, gradient=C.MOTION_COLORS['alphas'],
                            color=pred_fade)

                        lon_sn, lat_sn = (
                            pred[:, 1] - f * s[:, 1], pred[:, 0] - f * s[:, 0])
                        C.plot_faded_prediction(
                            lon_sn, lat_sn, movement_plot, gradient=C.MOTION_COLORS['alphas'],
                            color=pred_fade)

            all_lon.append(lon)
            all_lat.append(lat)

    # Plot movement
    movement_plot.set_xticks([])
    movement_plot.set_yticks([])

    if lat < abs(north) and lat > south and lon > west and lon < east:
        movement_plot.axis(
            [np.min(all_lon) - C.OFFSET, np.max(all_lon) + C.OFFSET,
             np.min(all_lat) - C.OFFSET, np.max(all_lat) + C.OFFSET])

    # Set figure bbox around the predicted trajectory
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{tag}.png', dpi=800)
    plt.close()


def plot_scene_marginal_with_hist(
    gt_history: np.array, pred_trajectories: np.array, pred_scores: np.array, sigmas: np.array,
    maps: Tuple, ll_extent: Tuple, gt_future: np.array = None, tag: str = 'temp.png', ego_id: int = 0,
    out_dir: str = './out'
) -> None:
    """ Function for visualizing the prediction of a scene. """
    mm = C.MOTION_COLORS['multi_modal']
    fade = colors.to_rgb(mm) + (0.0,)
    pred_fade = colors.LinearSegmentedColormap.from_list('my', [fade, mm])
    mm = (mm, 0.9)
    C.MOTION_COLORS['alphas'] = np.array(
        [0.15*((30-t)/20) for t in range(10, 30)])

    north, east, south, west = ll_extent

    # Save states
    fig, (movement_plot, speed_plot) = plt.subplots(
        2, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [3, 1]})

    bkg_ground, ac_map = maps[0], maps[1]
    # Display global map
    movement_plot.imshow(
        bkg_ground, zorder=0, extent=[west, east, south, north], alpha=1.0, cmap='gray_r')

    N, T, H, D = pred_trajectories.shape
    # Iterate through agent in the scene
    all_lon, all_lat, vel_sigma, vel_mu, scores = [], [], [], [], []

    zipped = zip(gt_history, gt_future, pred_scores,
                 pred_trajectories, sigmas[..., 0], sigmas[..., 1])
    for n, (gt_traj, pred_gt, hscores, preds, sigmas_p, sigmas_n) in enumerate(zipped):
        # Get heading at last point of trajectory.
        gt, heading = gt_traj[:, 1:], gt_traj[-1, 0]
        lon, lat = gt[-1, 1], gt[-1, 0]
        if lon == 0 or lat == 0:
            continue

        # Place plane on last point of ground truth sequence
        img = C.plot_agent(ac_map, heading)
        ab = AnnotationBbox(img, (lon, lat), frameon=False)
        movement_plot.add_artist(ab)

        # Plot past sequence
        movement_plot.plot(
            gt[:, 1], gt[:, 0], color=C.MOTION_COLORS['gt_hist'][0], lw=C.MOTION_COLORS['gt_hist'][1])
        movement_plot.plot(
            pred_gt[:, 1], pred_gt[:, 0], color=C.MOTION_COLORS['gt_future'][0],
            lw=C.MOTION_COLORS['gt_future'][1], linestyle='dashed')

        if n == ego_id:
            movement_plot.scatter(
                lon, lat, color=C.MOTION_COLORS['ego_agent'][0], alpha=C.MOTION_COLORS['ego_agent'][1],
                s=300)

            history_velocity = C.get_velocity(gt)
            future_velocity = C.get_velocity(pred_gt)

            # Iterate through prediction header
            for h in range(H):
                score = hscores[h]  # Get confidence score for head
                scores.append(score)

                if score > 0.02:
                    pred = preds[:, h]  # Get best sampled predicted trajectory
                    s_p = sigmas_p[:, h, :]  # Positive sigma
                    s_n = sigmas_n[:, h, :]  # Negative sigma
                    s = ((s_p - s_n) / 2)  # Sigma

                    predicted_velocity = C.get_velocity(pred[:])
                    vel_mu.append(predicted_velocity)

                    # Plot predicted trajectory.
                    movement_plot.plot(
                        pred[:, 1], pred[:, 0], color=C.MOTION_COLORS['pred'][0],
                        lw=C.MOTION_COLORS['pred'][1], alpha=score.item())

                    # Evaluate positive boundary for prediction distribution
                    positive_sigma = pred + s
                    vel_sp = C.get_velocity(positive_sigma)
                    vel_sigma.append(vel_sp)

                    # Evaluate negative boundary for prediction distribution
                    negative_sigma = pred - s
                    vel_sn = C.get_velocity(negative_sigma)
                    vel_sigma.append(vel_sn)

                    # Plot motion distribution
                    movement_plot.fill_between(
                        pred[:, 1], s_n[:, 0], s_p[:, 0], lw=mm[1], alpha=score.item() * 0.65,
                        color=mm[0], interpolate=True)

            all_lon.append(lon)
            all_lat.append(lat)

    # Plot movement
    movement_plot.set_xticks([])
    movement_plot.set_yticks([])

    if lat < abs(north) and lat > south and lon > west and lon < east:
        movement_plot.axis(
            [np.min(all_lon) - C.OFFSET, np.max(all_lon) + C.OFFSET,
             np.min(all_lat) - C.OFFSET, np.max(all_lat) + C.OFFSET])

    C.plot_speed_histogram(speed_plot, vel_mu, vel_sigma,
                           history_velocity, future_velocity)

    # Set figure bbox around the predicted trajectory
    plt.savefig(f'{out_dir}/{tag}.png', dpi=500, bbox_inches='tight')
    plt.close()
