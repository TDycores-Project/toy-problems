#!/bin/bash
declare -A run_flags
    run_flags[problem]=infsup
    run_flags[nelx]=2
    run_flags[nely]=2
    run_flags[mesh]=uniform

declare -A test_flags
    test_flags[res_start]=2
    test_flags[res_stride]=2
    test_flags[res_end]=10

fileName=InfSup.csv
echo ",mesh_res,infsup" > $fileName
i=0

for ((res=${test_flags[res_start]}; res<=${test_flags[res_end]}; res+=${test_flags[res_stride]})); do
    run_flags[nelx]=$res
    run_flags[nely]=$res
    args=''
    for arg in "${!run_flags[@]}"; do
        if ! [[ -z ${run_flags[$arg]} ]]; then
            args="$args --$arg ${run_flags[$arg]}"
        fi
    done
    python3 FEAC1red2D.py $args | grep "infsup constant:" | awk -v i="$i" -v res="$res" '{ print i","res","$3}' >> $fileName
    i=$((i+1))
done
