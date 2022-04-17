# Information
This project contains the code files to generate scientific figures in the paper.

Hsu, T. Y., Primeau, F. W., & Magnusdottir, G. (2022). A Hierarchy of Global Ocean Models Coupled to CESM1. Journal of Advances in Modeling Earth Systems.

# Requirement

- Platform  : Unix-like machine
- Languages : Python3 (Matplotlib, Scipy, Numpy, Cartopy, netCDF4), Julia (PyPlot, PyCall, NCDatasets, Formatting, Statistics, ArgMacros, Distributions)

# How to generate the figures

First, download the dataset from [https://doi.org/10.7280/D1V97T](https://doi.org/10.7280/D1V97T) or [Datadryad direct download link](https://datadryad.org/stash/share/fizzfQYLdKduFO_alkDIXkNM0qBbrGBXbolkMW64d6Y)

Then, put the unzipped dataset into the workspace. The arrangement should be as

```
workspace (plots-A-Hierarchy-of-Global-Ocean-Models-Coupled-to-CESM1)
│
├── CESM_domains
│   ├── atm_ilev.nc
│   ├── atm_lev.nc
│   └── domain.lnd.fv0.9x1.25_gx1v6.090309.nc
├── clean.sh
├── colordef.py
├── data
│   ├── AMOC
│   ├── hierarchy_average
│   ├── hierarchy_statistics
│   ├── supp
│   └── README
├── paper_figures
├── make_figures.sh
├── plot_fig02.py
├── plot_fig03.py
├── plot_fig04.py
├── plot_fig05.py
├── plot_fig06.jl
├── plot_fig07.py
├── plot_fig08.py
├── plot_fig09.py
├── plot_fig10.py
├── plot_fig11.py
├── plot_fig12_fig13.py
├── plot_fig14.py
├── quick_tools.py
├── remove_ENSO_annual.jl
└── SpectralAnalysis.jl
```

Finally, execute the following scripts to produce figures. The output figures will be in the newly created folder `figures`

```
$> ./01_make_figures.sh         # Generate figures
$> ./02_postprocess_figures.sh  # Trim the figures mostly
$> ls final_figures/            # List the generated figures
$> ./99_clean.sh         # This removes all the figures and meta-data (data_extra directory) generated
```
