# [AmeliaScenes](http://ameliacmu.github.io/)

AmeliaScenes is an automated method to select relevant ego-agents and produce diverse complex scene representations. It takes data generted by AmeliaSWIM and AmeliaMaps  to characterize relevant behaviours and interactions within airport operations, creatin a training, validation and testing splits.

## Overview

<!-- We use Amelia-Scenes to extract scenes from the raw CSV files. Our scene extraction is easily configurable to obtain different types of scenes in terms of number of agents of interest, scene length, and point-to-point granularity.

We also provide scene characterization tools to analyze scene complexity w.r.t individual agent kinematic profiles, as well as level of agent-to-agent interactivity and crowdedness. -->



<!-- Contributions
[Ingrid Navarro](https://navars.xyz) *, [Jay Patrikar](https://www.jaypatrikar.me) *, Joao P. A. Dantas,
Rohan Baijal, Ian Higgins, [Sebastian Scherer](https://theairlab.org/team/sebastian/) and [Jean Oh](https://www.cs.cmu.edu/~./jeanoh/) -->

## Pre-requisites

### Dataset

For the repository to work the dataset must be downloaded or linked as follows:

```bash
|-- datasets
    |-- amelia
        |-- assets
        |-- graph_data_a10v01os
        |-- traj_data_a10v08

```

## Installation

### Creating an environment

It can be created using either `conda`.

- Conda

```bash
conda create --name amelia python=3.9
conda activate amelia
```

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaScenes.git
cd AmeliaScenes
pip install -e .
```

### Dataset Processing


<!-- For scene processing   -->
With the `amelia` environment run:

```bash
python amelia_scenes/scene_processing/run_processor.py --airport [airport_name] --base_dir [path_to_dataset] --traj_version [version] --graph_version [version] --parallel [parallel] --overwite [overwite] --perc_process [percentage] --to_process [to_process] --seed [seed]
```

Where:

- `[airport]`: Name of the airport to be processed. It can be one of the following: `kbos`, `kdca`, `kewr`, `kjfk`, `klax`, `kmdw`, `kmsy`, `ksea`, `ksfo`, `panc`. By default it is set to `kbos`.
- `[base_dir]`: Path to the dataset. By defaulult the path is set to `../../datasets/amelia`.
- `[traj_version]`: Version of the trajectory data. By default it is set to `a10v08`.
- `[graph_version]`: Version of the graph data. By default it is set to `a10v01os`.
- `[parallel]`: If the processing should be done in parallel. By default it is set to `True`.
- `[overwrite]`: If the processing should overwrite the existing data. By default it is set to `True`.
- `[perc_process]`: Top limit vizualization of the data being processed. By default it is set to `1.0`.
- `[to_process]`: Type of scene to be processed. It may be `scenes`, `metas` or `both`.  By default it is set to `both`.
- `[seed]`: Seed for the random number generator. By default it is set to `42`.

``` bash
python amelia_scenes/scene_splitting/create_splits.py --split_type [split_type] --base_dir [path_to_dataset] --traj_version [version] --graph_version [version] --seed [seed]
```

Where:

```


### To do
- do create splits
- do overview
- meta -> other kinetic data
scene processe raw data from csv
- Prerequisits




### Results

When running the code the following files will be generated:
