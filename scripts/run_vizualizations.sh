#!/bin/bash

conda activate amelia

cd amelia/AmeliaScenes

airports=(
    "kbos KBOS_0_203_1728079864 001362_n-29"
    "kdca KDCA_0_12_1686573191 001224_n-18"
    "kjfk KJFK_0_48_1734145992 000278_n-36"
    "klax KLAX_0_109_1725777763 001074_n-37"
    "kpit KPIT_0_19_1712914978 001726_n-26"
    "ksea KSEA_0_134_1688479289 002312_n-24"
    "ksfo KSFO_0_54_1739776165 000153_n-17"
    "panc PANC_0_116_1703270477 003089_n-23"
)




for airport in "${airports[@]}"; do
    airport_name=$(echo $airport | cut -d' ' -f1)
    scene=$(echo $airport | cut -d' ' -f2)
    scene_id=$(echo $airport | cut -d' ' -f3)
    echo "Processing airport: $airport_name, scene: $scene, scene_id: $scene_id"
    python amelia_scenes/viz_scenes.py --traj_version a42v99percentile --graph_version a42v01os --perc 1.0 --scene_type gif  --dpi 400 --airport ${airport_name} --show-scores --viz_scene ${scene} --scene_id ${scene_id}
done
