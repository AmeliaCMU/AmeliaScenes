# Amelia Framework

## Installation

Installation can be done by installing the AmeliaScenes repository or the complete Amelia Framework

### Creating an environment

It can be created using either `conda`.

- Conda

```bash
conda create --name amelia python=3.9
conda activate amelia
```

## Installing AmeliaScenes

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaScenes.git
cd AmeliaScenes
pip install -e .
```

## Installing the Amelia Framework

Make sure that you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed and download the `install.sh` script from the repository and run:

```bash
chmod +x install.sh
./install.sh amelia
```
