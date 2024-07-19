# import cv2
# import imageio.v2 as imageio
# import json
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import torch 

# from easydict import EasyDict
# from geographiclib.geodesic import Geodesic
# from math import radians, sin, cos
# from matplotlib import rcParams
# from matplotlib.collections import LineCollection
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from scipy import ndimage
# from torch import tensor
# from typing import Tuple, List

# from src.utils import global_masks as G
# from src.utils.utils import get_velocity, xy_to_ll

# def wrap_angle(angle):
#     return np.radians(((angle % 360) + 540) % 360 - 180)

# rcParams['font.family'] = 'monospace'

# def lpfilter(input_signal, win):
#     # Low-pass linear Filter
#     # (2*win)+1 is the size of the window that determines the values that influence 
#     # the filtered result, centred over the current measurement
#     from scipy import ndimage
#     kernel = np.lib.pad(np.linspace(1,3,win), (0,win-1), 'reflect') 
#     kernel = np.divide(kernel,np.sum(kernel)) # normalise
#     kernel = np.stack((kernel,kernel),axis=1)
#     output_signal = ndimage.convolve(input_signal, kernel) 
#     return output_signal

# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float,axis=0)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

# def save_scene_info(
#     gt_history: tensor, mu: tensor, pred_trajectories: tensor, pred_scores: tensor, sigmas: tensor, 
#     context_patch: tensor, X: tensor, hist_abs: tensor, hist_rel: tensor, tag: str
# ) -> None:
#     np.savez(
#         f'out/info/{tag}.npz', 
#         X.detach().cpu().numpy(),
#         hist_abs.detach().cpu().numpy(), 
#         hist_rel.detach().cpu().numpy(), 
#         context_patch.detach().cpu().numpy(), 
#         gt_history.detach().cpu().numpy(), 
#         mu.detach().cpu().numpy(), 
#         pred_trajectories.detach().cpu().numpy(), 
#         pred_scores.detach().cpu().numpy(), 
#         sigmas.detach().cpu().numpy()
#     )

# def plot_speed_histogram(ax, vel_mu, vel_sigma, vel_hist, vel_fut):
#     ax.set_xlim([0,140])
        
#     #Plot velocity distribution
#     freq, _, patches = ax.hist(
#         vel_sigma, bins = 10, color= MOTION_COLORS['multi_modal'][0], edgecolor = "k", 
#         linewidth = 0.5, alpha = 1, density = True)
#     ax.axvline(vel_hist, 0, 1000, color = MOTION_COLORS['gt_hist'][0], linewidth = 1.9)
#     ax.axvline(vel_fut, 0, 1000, color = MOTION_COLORS['gt_future'][0], linewidth = 1.9)
#     ax.text(
#         vel_hist, max(freq), "H", rotation = 90, verticalalignment = 'center', 
#         color = MOTION_COLORS['gt_hist'][0])
#     ax.text(
#         vel_fut, max(freq), "G", rotation = 90, verticalalignment = 'center', 
#         color = MOTION_COLORS['gt_future'][0])
    
#     # Iterate through predicted speed
#     for i, v in enumerate(vel_mu):
#         ax.axvline(v, 0, 1000, color = MOTION_COLORS['pred'][0], linewidth = 1.9)
        
#     ax.set_yticks([])
#     ax.set_xticks([0, 35 , 70, 105, 140])
#     ax.set_xlabel("Knots")
#     ax.set_title("Speed Histogram", fontdict = title_font)
#     ax.grid(True)

# def plot_scene_joint(
#     gt_history: tensor, pred_trajectories: tensor, pred_scores: tensor, sigmas: tensor, maps: Tuple, 
#     ll_extent: Tuple, gt_future: np.array = None, tag: str = 'temp.png', ego_id: int = 0 , 
#     fast_plot: bool = False, out_dir: str = './out'
# ) -> None:
#     mm =  (20/255, 245/255, 178/255, 1.0)
#     fade = colors.to_rgb(mm) + (0.0,)
#     pred_fade = colors.LinearSegmentedColormap.from_list('my', [fade, mm])
#     mm = (mm, 0.9)
#     MOTION_COLORS['alphas'] = np.array([0.15*((30-t)/20) for t in range(10,30)])

#     north, east, south, west = ll_extent

#     # Convert tensors to numpy arrays
#     pred_scores = pred_scores.detach().cpu().numpy()
#     gt_future = gt_future.detach().cpu().numpy()
#     pred_trajectories = pred_trajectories.detach().cpu().numpy()
#     sigmas = sigmas.detach().cpu().numpy()
#     gt_history = gt_history.detach().cpu().numpy()   
    
#     # Get head scores
#     error = np.linalg.norm((pred_trajectories - gt_future[..., None, :]), axis=-1)
#     error = (error * pred_scores[..., None, :] + EPS).mean(axis=(0, 1))
#     scores = np.clip((error - error.min()) / (error.max() - error.min()), 0.1, 1.0)

#     # Save states
#     if (ego_id != -1): 
#         fig, (movement_plot, speed_plot) = plt.subplots(
#             2, 1, figsize = (6,9), gridspec_kw = {'height_ratios': [3, 1]})
#     else:  
#         fig, movement_plot = plt.subplots()
    
#     bkg_ground, ac_map = maps[0], maps[1]
#     # Display global map
#     movement_plot.imshow(
#         bkg_ground, zorder=0, extent=[west, east, south, north], alpha=1.0, cmap='gray_r') 
    
#     N, T, H, D = pred_trajectories.shape 
#     # Iterate through agent in the scene
#     all_lon, all_lat, vel_sigma, vel_mu = [], [], [], []
    
#     zipped = zip(gt_history, gt_future, pred_trajectories, sigmas[..., 0], sigmas[..., 1])
#     for n, (gt_traj, pred_gt, preds, sigmas_p, sigmas_n) in enumerate(zipped):
#         gt, heading = gt_traj[:, 1:], gt_traj[-1, 0] # Get heading at last point of trajectory.
#         lon, lat = gt[-1, 1], gt[-1, 0]
#         if lon == 0 or lat == 0:
#             continue
        
#         # Place plane on last point of ground truth sequence
#         img = plot_agent(ac_map, heading)
#         ab = AnnotationBbox(img, (lon, lat), frameon = False) 
#         movement_plot.add_artist(ab)
        
#         # Plot past sequence
#         movement_plot.plot(
#             gt[:, 1], gt[:, 0], color = MOTION_COLORS['gt_hist'][0], lw = MOTION_COLORS['gt_hist'][1]) 
#         movement_plot.plot(
#             pred_gt[:, 1], pred_gt[:,0], color = MOTION_COLORS['gt_future'][0], 
#             lw = MOTION_COLORS['gt_future'][1], linestyle = 'dashed')
        
#         if n == ego_id: 
#             movement_plot.scatter(
#                 lon, lat ,color=MOTION_COLORS['ego_agent'][0], alpha=MOTION_COLORS['ego_agent'][1], 
#                 s = 300)
#             history_velocity = get_velocity(gt)
#             future_velocity = get_velocity(pred_gt)

#         # Iterate through prediction header
#         for h in range(H):
#             score = scores[h]
#             if score > 0.02:
#                 pred = preds[:, h] # Get best sampled predicted trajectory
#                 s_p = sigmas_p[:, h, :] # Positive sigma
#                 s_n = sigmas_n[:, h, :] # Negative sigma
#                 s = ((s_p - s_n) / 2) # Sigma
                
#                 if n == ego_id:           
#                     predicted_velocity = get_velocity(pred[:])
#                     vel_mu.append(predicted_velocity)
                
#                 #Plot predicted trajectory.
#                 movement_plot.plot(
#                     pred[:, 1], pred[:, 0], color = MOTION_COLORS['pred'][0], 
#                     lw = MOTION_COLORS['pred'][1], alpha=score)  
                
#                 if fast_plot:
#                     # Evaluate positive boundary for prediction distribution 
#                     positive_sigma = pred + s
#                     vel_sp = get_velocity(positive_sigma)
#                     if n == ego_id: 
#                         vel_sigma.append(vel_sp)
                    
#                     # Evaluate negative boundary for prediction distribution
#                     negative_sigma = pred - s 
#                     vel_sn = get_velocity(negative_sigma)
#                     if n == ego_id: 
#                         vel_sigma.append(vel_sn)

#                     # Plot motion distribution
#                     movement_plot.fill_between(
#                         pred[:,1], s_n[:,0], s_p[:,0], lw = mm[1], alpha = score * 0.65, 
#                         color = mm[0], interpolate = True)
#                 else:
#                     # Add other samples based on the confidence of the prediction (with a max of 10)
#                     samples = int(score * 500)
#                     for _ in range(samples):
#                         f = abs(np.random.normal(0,0.3)) # Get normal distribution
#                         # Evaluate speed for positive sigma distribution
#                         positive_sigma = (pred + f * s )
#                         vel_sp = get_velocity(positive_sigma)
#                         if n == ego_id: 
#                             vel_sigma.append(vel_sp)

#                         lon_sp, lat_sp = (pred[:, 1] + f * s[:, 1], pred[:, 0] + f * s[:, 0])
#                         plot_faded_prediction(
#                             lon_sp, lat_sp, movement_plot, gradient = MOTION_COLORS['alphas'], 
#                             color = pred_fade)
                        
#                         # Evaluate speed for negative sigma distribution
#                         negative_sigma = (pred - f * s)
#                         vel_sn = get_velocity(negative_sigma)
#                         if n == ego_id: 
#                             vel_sigma.append(vel_sn)

#                         lon_sn, lat_sn = (pred[:, 1] - f * s[:, 1], pred[:, 0] - f * s[:, 0])
#                         plot_faded_prediction(
#                             lon_sn, lat_sn, movement_plot, gradient = MOTION_COLORS['alphas'], 
#                             color = pred_fade)

#             all_lon.append(lon)
#             all_lat.append(lat)

#     # Plot movement
#     movement_plot.set_xticks([])
#     movement_plot.set_yticks([])
    
#     if lat < abs(north) and lat > south and lon > west and lon < east:
#         movement_plot.axis(
#             [np.min(all_lon) - OFFSET, np.max(all_lon) + OFFSET, 
#              np.min(all_lat) - OFFSET, np.max(all_lat) + OFFSET])
   
#     if ego_id != -1:
#         plot_speed_histogram(speed_plot, vel_mu, vel_sigma, history_velocity, future_velocity)
    
#     # Set figure bbox around the predicted trajectory
#     plt.show(block = False)
#     plt.savefig(f'{out_dir}/{tag}.png',dpi=500)
#     plt.close()
