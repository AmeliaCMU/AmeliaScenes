# [AmeliaScenes](http://ameliacmu.github.io/)




<!-- Contributions
[Ingrid Navarro](https://navars.xyz) *, [Jay Patrikar](https://www.jaypatrikar.me) *, Joao P. A. Dantas,
Rohan Baijal, Ian Higgins, [Sebastian Scherer](https://theairlab.org/team/sebastian/) and [Jean Oh](https://www.cs.cmu.edu/~./jeanoh/) -->

## Installation

### Creating an environment

It can be created using either `conda` or `virtualenvwrapper`.

- Conda

```bash
conda create --name amelia python=3.9
conda activate amelia
```

- virtualenvwrapper

```bash
mkvirtualenv amelia -p python3.9
workon amelia
```

Download the repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaScenes.git
cd AmeliaScenes
pip install -e .
```

By ruuning the previos comend the following files will be generated:

--- set files


### Dowloading the Dataset

--- How is this going to be handled?


### Running the code

With the created environment run:

```bash
python amelia_scenes/scene_processing/run_processor.py --airport [airport_name] --base_dir [path_to_dataset] --traj_version [version] --graph_version [version] --parallel [] --overwite [] --perc_process [percentage] --to_process [to_process] --seed [seed]
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

### Results

When running the code the following files will be generated:
