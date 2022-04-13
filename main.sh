#!/bin/bash

py=python3
ju=julia


plot_codes=(
    $py plot_fig02.py
    $py plot_fig12_fig13.py
)

plot_codes=(
    $py plot_variability_SST_col.py
)


# Some code to download data and extract them

mkdir -p data_extra

echo "Making extra data into data_extra"
#$ju remove_ENSO_annual.jl


for i in seq 1 $(( ${#plot_codes[@]} / 2 )) ; do
    PROG="${plot_codes[$(( (i-1) * 2 + 0 ))]}"
    FILE="${plot_codes[$(( (i-1) * 2 + 1 ))]}"
    echo "=====[ Running file: $FILE ]====="
    eval "$PROG $FILE"
done
