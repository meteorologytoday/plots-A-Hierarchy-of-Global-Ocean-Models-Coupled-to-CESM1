import cartopy.crs as ccrs
import matplotlib as mplt
from matplotlib import cm
import matplotlib.patches as patches

import os

#mplt.use('Agg')

from matplotlib import rc

default_linewidth = 2.0;
default_ticksize = 10.0;

mplt.rcParams['lines.linewidth'] =   default_linewidth;
mplt.rcParams['axes.linewidth'] =    default_linewidth;
mplt.rcParams['xtick.major.size'] =  default_ticksize;
mplt.rcParams['xtick.major.width'] = default_linewidth;
mplt.rcParams['ytick.major.size'] =  default_ticksize;
mplt.rcParams['ytick.major.width'] = default_linewidth;

#rc('font', **{'family':'sans-serif', 'serif': 'Bitstream Vera Serif', 'sans-serif': 'MS Reference Sans Serif', 'size': 20.0});
rc('font', **{'size': 12.0});
rc('axes', **{'labelsize': 12.0});
rc('mathtext', **{'fontset':'stixsans'});
#rc(('xtick.major','ytick.major'), pad=20)

#import matplotlib.font_manager as fm;
#print("%s: %d"%(fm.FontProperties().get_name(),fm.FontProperties().get_weight()));

import matplotlib.pyplot as plt

import sys, argparse
from netCDF4 import Dataset
import numpy as np
from pprint import pprint



proj1 = ccrs.PlateCarree(central_longitude=180.0)
data_proj = ccrs.PlateCarree(central_longitude=0.0)

proj_kw = {
    'projection':proj1,
    'aspect' : 'auto',
}


print("Creating figure object...")
fig, ax = plt.subplots(1, 1, figsize=(6, 8), constrained_layout=True, subplot_kw=proj_kw)
print("Created.")
            
boxes = [
    ["Box1", [-60, 60], [150, 250], "r"],
    ["Box2", [-60, 60], [300, 355], "r"],
]

for box_label, lat_rng, lon_rng, linecolor in boxes:
    ax.add_patch(patches.Rectangle((lon_rng[0], lat_rng[0]), lon_rng[1] - lon_rng[0], lat_rng[1] - lat_rng[0], linewidth=1, edgecolor=linecolor, facecolor='none', transform=data_proj))

ax.coastlines()

ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
plt.show()

