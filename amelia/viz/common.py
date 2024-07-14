import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage
from scipy import ndimage
from pyproj import Transformer


# -------------------------------------------------------------------------------------------------#
#                                   COMMON GLOBAL VARIABLES                                        #
# -------------------------------------------------------------------------------------------------#
MAP_CRS = 'EPSG:4326'

STYLES = {
    'family': 'monospace', 
    'color':  'black', 
    'weight': 'bold',
}

OFFSET = 0.0073
EPS = 1e-5

COLOR_CODES = {
    1: '#ff9b21',
    2: '#fc0349',
    3: '#1fe984',
    4: '#6285cf',
    5: '#f05eee',
}

MOTION_COLORS = {
    'gt_hist': ('#FF5A4C', 0.9 ),
    'gt_future': ('#8B5FBF', 0.9),
    'multimodal': ('#14F5B2', 1.0),
    'multi_modal': ((20/255, 245/255, 178/255), 1.0),
    'pred': ('#07563F', 0.9),
    'ego_agent': ('#0F94D2', 0.5),
}

AIRCRAFT = 0
VEHICLE = 1
UNKNOWN = 2

KNOTS_TO_MPS = 0.51444445
KNOTS_TO_KPH = 1.852
HOUR_TO_SECOND = 3600
KMH_TO_MS = 1/3.6

ZOOM = {
    AIRCRAFT: 0.015, 
    VEHICLE: 0.2, 
    UNKNOWN: 0.015
}

LL_TO_KNOTS = 1000 * 111 * 1.94384

# -------------------------------------------------------------------------------------------------#
#                                      COMMON PLOT UTILS                                           #
# -------------------------------------------------------------------------------------------------#
def plot_agent(asset, heading, zoom = 0.015):
    img = ndimage.rotate(asset, heading) 
    img = np.fliplr(img) 
    img = OffsetImage(img, zoom=zoom)
    return img

def plot_faded_prediction(lon: np.array, lat: np.array, axis = None, gradient = None, color = None):
    points = np.vstack((lon, lat)).T.reshape(-1, 1, 2)
    segments = np.hstack((points[:-1], points[1:]))
    lc = LineCollection(segments, lw = 0.35, array = gradient, cmap = color)
    axis.add_collection(lc)
    del lc

def get_velocity(trajectory: np.array):
    # Calculate displacement vector
    displacement = trajectory[1:] - trajectory[0:-1]
    # Use norm to get magnitude of displacement and take mean to get speed
    speed = np.linalg.norm(displacement, axis=1).mean() * LL_TO_KNOTS
    return speed

def plot_speed_histogram(ax, vel_mu, vel_sigma, vel_hist, vel_fut):
    ax.set_xlim([0,140])
        
    #Plot velocity distribution
    freq, _, patches = ax.hist(
        vel_sigma, bins = 10, color= MOTION_COLORS['multi_modal'][0], edgecolor = "k", 
        linewidth = 0.5, alpha = 1, density = True)
    ax.axvline(vel_hist, 0, 1000, color = MOTION_COLORS['gt_hist'][0], linewidth = 1.9)
    ax.axvline(vel_fut, 0, 1000, color = MOTION_COLORS['gt_future'][0], linewidth = 1.9)
    ax.text(
        vel_hist, max(freq), "H", rotation = 90, verticalalignment = 'center', 
        color = MOTION_COLORS['gt_hist'][0])
    ax.text(
        vel_fut, max(freq), "G", rotation = 90, verticalalignment = 'center', 
        color = MOTION_COLORS['gt_future'][0])
    
    # Iterate through predicted speed
    for i, v in enumerate(vel_mu):
        ax.axvline(v, 0, 1000, color = MOTION_COLORS['pred'][0], linewidth = 1.9)
        
    ax.set_yticks([])
    ax.set_xticks([0, 35 , 70, 105, 140])
    ax.set_xlabel("Knots")
    ax.set_title("Speed Histogram", fontdict = STYLES)
    ax.grid(True)
    
def transform_extent(extent, original_crs: str, target_crs: str):
    transformer = Transformer.from_crs(original_crs, target_crs)
    north, east, south, west = extent
    xmin_trans, ymin_trans = transformer.transform(south, west)
    xmax_trans, ymax_trans = transformer.transform(north, east)
    return (ymax_trans, xmax_trans, ymin_trans, xmin_trans)

def reproject_sequences(sequence, target_projection):
    lat = np.array(sequence[:,0::2])
    lon = np.array(sequence[:,1::2])
    projector = Transformer.from_crs(MAP_CRS, target_projection)
    x, y = projector.transform(lat, lon)
    transformed_sequence = []
    for x, y in zip(x, y):
        transformed_sequence.append(y)
        transformed_sequence.append(x)
    
    transformed_sequence = np.array(transformed_sequence)
    T, D = sequence.shape
    
    transformed_sequence = transformed_sequence.reshape((T,D))
    return transformed_sequence