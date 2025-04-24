import matplotlib.pyplot as plt

from typing import Tuple

from amelia_scenes.visualization import common as C
from amelia_scenes.visualization import benchmark_viz as bench
from amelia_scenes.visualization import scoring_viz as scoring
from amelia_scenes.visualization import marginal_predictions as marginal
from amelia_scenes.utils import global_masks as G


# available scene visualization
SUPPORTED_SCENES_TYPES = [
    'simple',
    'benchmark',
    'benchmark_pred',
    'marginal_pred',
    'joint_pred',
    'features',
    'scores',
    'strategy'
]


def plot_scene(
    scene: dict,
    assets: Tuple,
    filename: str,
    scene_type: str,
    dpi: int = 200,
    **kwargs
):
    # predictions: Tuple = None,
    # scores: bool = False,
    # features_to_add: list = [],
    # features: dict = {},
    """ Wrapper for plotting scenes.

    Inputs
    ------
        scenario[dict]: dictionary containing the scene components.
        assets[Tuple]: tuple containing all scene assets (e.g., map, graph, agent assets, etc.)
        filetag[str]: nametag for the image to save.
        features_to_add[list]: list of features to create subplots for.
        features[dict]: computed features for each agent.
        dpi[int]: image dpi to save.
    """
    assert scene_type in SUPPORTED_SCENES_TYPES, f"Scene type not supported {scene_type}"

    reproject = True if scene['airport_id'] in ['panc', 'kmsy'] else False
    if scene_type == 'simple':
        agents = [] if kwargs.get('agents_interest') is None else kwargs.get('agents_interest')
        plot_scene_simple(scene, assets, filename, dpi, reproject=reproject, agents_interest=agents)
    elif scene_type == 'benchmark':
        benchmark = scene['benchmark']
        bench.plot_scene_benchmark(scene, assets, benchmark, filename, dpi, reproject=reproject)
    elif scene_type == 'benchmark_pred':
        predictions = kwargs.get('predictions')
        assert predictions, "Predictions not provided" 
        benchmark = scene['benchmark']
        # benchmark = None
        bench.plot_scene_benchmark_predictions(
            scene, assets, benchmark, predictions, filename, dpi, reproject=reproject)
    elif scene_type == 'marginal_pred':
        predictions = kwargs.get('predictions')
        assert predictions, "Predictions not provided"
        plot_all = kwargs.get('plot_all')
        assert plot_all, "Plot all agents not provided"
        marginal.plot_scene_marginal(
            scene, assets, predictions, filename, dpi, reproject=reproject, plot_all=plot_all)

    elif scene_type == 'scores':
        # raise NotImplementedError
        # agents_interest = benchmark['bench_agents']
        agent_sequences, agent_masks = scene['agent_sequences'][:, :, G.HLL], scene['agent_masks']
        agent_types, agent_ids = scene['agent_types'], scene['agent_ids']
        agent_scores    = scene['meta']['agent_scores'] 
        # if scores else None
        scoring.plot_scene_scores(
            agent_sequences, agent_scores, assets, agent_masks, agent_types, agent_ids, tag=filename, dpi=dpi
            )
    else:
        raise NotImplementedError
    # elif scene_type == 'strategy':
    #     raise NotImplementedError
    #     agent_order = scenario['meta']['agent_order']
    #     scoring.plot_scene_strategy(
    #         agent_sequences, agent_order, assets, agent_masks, agent_types, agent_ids, tag=filetag,
    #         dpi=dpi)
    # else:
    #     raise NotImplementedError
    #     scoring.plot_scene_features(scenario, assets, filetag, features_to_add, features, dpi=dpi)


def plot_scene_simple(
    scene: dict, assets: Tuple, filename: str = 'temp.png', dpi=600, agents_interest: list = [],
    reproject: bool = False, projection: str = 'EPSG:3857'
) -> None:
    """ Visualize simple scenes """
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    fig, ax = plt.subplots()
    # breakpoint()
    # Display global map
<<<<<<< Updated upstream

    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.3)

=======
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.3) 
>>>>>>> Stashed changes
    C.plot_sequences(
        ax, scene, agents, agents_interest=agents_interest, reproject=reproject, projection=projection)
    C.save(ax, filename, dpi)#, limits=[west, east, south, north])
