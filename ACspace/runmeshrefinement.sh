#!/bin/bash
declare -A run_flags
    run_flags[problem]=convrate
    # GAUSS is AC, LGL is WY
    run_flags[massQmode]=GAUSS
    run_flags[nelx]=4
    run_flags[nely]=4
    run_flags[MMS]=quartic
    run_flags[mesh]=random

declare -A test_flags
    test_flags[res_start]=4
    test_flags[res_stride]=4
    test_flags[res_end]=16

echo "run,h,u_error,p_error" > ConvResult.csv
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
    python3 FEAC1red2D.py $args | grep "h, Velocity and Pressure Absolute Error:" | awk -v i="$i" '{ print i","$7","$8","$9}' >> ConvResult.csv
    i=$((i+1))
done
