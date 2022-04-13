#!/bin/bash

py=python3
ju=julia

# Some code to download data and extract them


mkdir -p data_extra

$ju remove_ENSO_annual.jl
$py plot_variability_SST_seasons_col.py 
