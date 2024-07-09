import json
import os
import pickle

import amelia.scenes.utils.common as C

from easydict import EasyDict
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Tuple, List

from amelia.scenes.scoring.crowdedness import compute_simple_scene_crowdedness
from amelia.scenes.scoring.kinematic import compute_kinematic_scores
from amelia.scenes.scoring.interactive import compute_interactive_scores
from amelia.scenes.scoring.critical import compute_simple_scene_critical


class SceneMetaProcessor:
    """ Dataset class for pre-processing meta data for airport surface movement scenes. Assumes that
    the data has been pickled already, i.e., need to run run_processor.py in scene mode first.
    """

    def __init__(self, config: EasyDict) -> None:
        """
        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the
            airport trajectory data.
        """
        super(SceneMetaProcessor, self).__init__()
        # Trajectory configuration
        self.airport = config.airport
        self.in_data_dir = os.path.join(config.in_data_dir, self.airport)
        self.out_data_dir = os.path.join(config.out_data_dir, self.airport)
        os.makedirs(self.out_data_dir, exist_ok=True)

        self.parallel = config.parallel
        self.overwrite = config.overwrite
        self.seed = config.seed

        self.pred_lens = config.pred_lens
        self.pred_len = max(self.pred_lens)
        self.hist_len = config.hist_len
        self.seq_len = self.hist_len + self.pred_len
        self.skip = config.skip
        self.min_agents = config.min_agents
        self.max_agents = config.max_agents
        self.min_valid_points = config.min_valid_points

        limits_file = os.path.join(
            config.assets_dir, self.airport, 'limits.json')
        with open(limits_file, 'r') as fp:
            self.ref_data = EasyDict(json.load(fp))

        graph_data_dir = os.path.join(config.graph_data_dir, self.airport)
        print(f"Loading graph data from: {graph_data_dir}")
        pickle_map_filepath = os.path.join(
            graph_data_dir, "semantic_graph.pkl")
        with open(pickle_map_filepath, 'rb') as f:
            graph_pickle = pickle.load(f)
            self.hold_lines = graph_pickle['hold_lines']

        # NOTE: there is a scenario folder for each CSV file that was processed.
        self.data_files = []
        for _dir in os.listdir(self.in_data_dir):
            data_dir = os.path.join(self.in_data_dir, _dir)
            self.data_files += [os.path.join(data_dir, f)
                                for f in os.listdir(data_dir)]

            out_data_dir = os.path.join(self.out_data_dir, _dir)
            os.makedirs(out_data_dir, exist_ok=True)
        print(f"Processing meta data for airport {self.airport.upper()}.")

    def process_data(self) -> None:
        """ It will process the CSV data containing airport trajectory information and create shards
        containing scenario-level pickle data. If parallel is True, it  will process the data in
        parallel, otherwise it will do it sequentially. Once the sharding is done, it will return a
        list containing all generated scenarios and will blacklist any CSV files that failed to
        generate scenarios for the specified conditions.
        """
        # TODO: debug parallel processing
        if self.parallel:
            Parallel(n_jobs=32)(delayed(self.process_file)(f)
                                for f in tqdm(self.data_files))
        else:
            for f in tqdm(self.data_files):
                self.process_file(f)

    def process_file(self, f: str) -> Tuple[List, List, List, List, List, List]:
        """ Processes a single data file. It first obtains the number of possible sequences (given
        the parameters in the configuration file) and then generates scene-level pickle files with
        the corresponding scene's information.

        Inputs
        ------
            f[str]: name of the file to shard.
        """
        fsplit = f.split('/')
        base_time_stamp, scenario_id = fsplit[-2], fsplit[-1]
        with open(f, 'rb') as fp:
            scene = EasyDict(pickle.load(fp))

        scene_meta_filepath = os.path.join(
            self.out_data_dir, base_time_stamp, scenario_id)
        if self.overwrite or not os.path.exists(scene_meta_filepath):

            crowd_scene_score = compute_simple_scene_crowdedness(
                scene, self.max_agents)
            kin_agents_scores, kin_scene_score = compute_kinematic_scores(
                scene, self.hold_lines)
            int_agents_scores, int_scene_score = compute_interactive_scores(
                scene, self.hold_lines)
            crit_agent_scores, crit_scene_score = compute_simple_scene_critical(
                agent_scores_list=[kin_agents_scores, int_agents_scores],
                scene_score_list=[crowd_scene_score,
                                  kin_scene_score, int_scene_score]
            )

            scene_meta = {
                'num_agents': scene.num_agents,
                'base_timestamp': base_time_stamp,
                'scenario_id': scene.scenario_id,
                'airport_id': scene.airport_id,
                'agent_scores': {
                    'kinematic': kin_agents_scores,
                    'interactive': int_agents_scores,
                    'critical': crit_agent_scores
                },
                'agent_order': {
                    'random': C.get_random_order(scene.num_agents, scene.agent_valid, self.seed),
                    'kinematic': C.get_sorted_order(kin_agents_scores),
                    'interactive': C.get_sorted_order(int_agents_scores),
                    'critical': C.get_sorted_order(crit_agent_scores)
                },
                'scene_scores': {
                    'crowdedness': crowd_scene_score,
                    'kinematic': kin_scene_score,
                    'interactive': int_scene_score,
                    'critical': crit_scene_score
                },
            }

            with open(scene_meta_filepath, 'wb') as f:
                pickle.dump(scene_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
