#!/bin/bash

# sync.sh - Sync folders using rsync
USRNAME="canovill"
SRC_BASE="/ocean/projects/cis240030p/canovill/data/Amelia42-Mini/data/traj_data_a42v01/proc_scenes_percentile/99/"
DEST_BASE="/Volumes/acanvil/datasets/percentiles/99/"

# folders=(
#     kpdx kphx kpvd ksdf ksfo ksna panc
#     kbdl kbwi kclt kden kdtw kfll kiad kjfk klax kmci kmdw kmia kmsp kord kphl kpit ksan ksea kslc kstl phnl
# )

# To avoid password prompts, set up SSH key-based authentication.
# See: https://www.ssh.com/academy/ssh/keygen

# for folder in "${folders[@]}"; do
rsync -rltpDvp -e 'ssh -l '${USRNAME}'' data.bridges2.psc.edu:${SRC_BASE}  ${DEST_BASE}
# done
