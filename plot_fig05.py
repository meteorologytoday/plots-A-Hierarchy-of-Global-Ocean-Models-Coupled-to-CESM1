

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib as mplt
import matplotlib.ticker as mticker
from matplotlib import cm
from quick_tools import *

import os

#mplt.use('Agg')
#mplt.use('TkAgg')

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


domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "figures"

mode_selection = 4

EOF_MODE = ["ENSO", "AO", "AAO", "NAO", "PDO"][mode_selection]
sparsity = [1, 3, 3, 1, 1][mode_selection]
beg_idx = sparsity - 1
casenames = ["OGCM", "EMOM", "MLM", "SOM"]
sim_casenames = getSimcases(casenames)

modes = 4
plotted_modes = 1





print("Plotted EOF_MODE : %s" % EOF_MODE)

with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]

    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

    lat_sparse = lat[beg_idx::sparsity]
    lon_sparse = lon[beg_idx::sparsity]


data = {}

for scenario in ["CTL",]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():

        casename = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        filename = "data/hierarchy_statistics/%s/atm_analysis_%s.nc" % (casename, EOF_MODE)
            
        with Dataset(filename, "r") as f:
            print("Loading %s ..." % (casename, ))
            data[scenario][exp_name]["PCAs"] = f.variables["PCAs"][:]
            data[scenario][exp_name]["PCAs_ts"] = f.variables["PCAs_ts"][:]
            data[scenario][exp_name]["PCAs_ts_var"] = np.var(f.variables["PCAs_ts"][:], axis=0, keepdims=False)
            #print(data[scenario][exp_name]["PCAs_ts_var"])
            


try: 
    os.makedirs(output_dir)

except:
    pass

extent = None
flip = [None] * modes  # 1 or -1 to flip the sign of EOF

if EOF_MODE == "ENSO":
    proj1 = ccrs.PlateCarree(central_longitude=180.0)
    aspect = 2.0
    data_proj = ccrs.PlateCarree(central_longitude=0.0)
elif EOF_MODE == "AAO":
    proj1 = ccrs.SouthPolarStereo()
    aspect = "equal"
    data_proj = ccrs.PlateCarree()
elif EOF_MODE == "AO":
    proj1 = ccrs.NorthPolarStereo()
    aspect = "equal"
    data_proj = ccrs.PlateCarree()
    extent = [-180, 180, 30, 90]
elif EOF_MODE == "NAO":
    proj1 = ccrs.NorthPolarStereo()
    aspect = "equal"
    data_proj = ccrs.PlateCarree()
    extent = [-180, 180, 30, 90]
elif EOF_MODE == "PDO":
    proj1 = ccrs.PlateCarree(central_longitude=180.0)
    aspect = 2.0
    data_proj = ccrs.PlateCarree(central_longitude=0.0)
    extent = [110, 250, 20, 70]
    flip[0] = [1, -1, 1, -1]
else:
    raise Exception("Unknown EOF_MODE: %s" % EOF_MODE)

proj_kw = {
    'projection':proj1,
    'aspect' : aspect,
}




# Original
fig = plt.figure(figsize=(6*plotted_modes, 3*len(sim_casenames)), constrained_layout=True)
#fig = plt.figure()
heights  = [1] * len(sim_casenames)
widths = [1,] * plotted_modes
spec = fig.add_gridspec(nrows=len(sim_casenames), ncols=plotted_modes, wspace=0.2, hspace=0.2) 

fig.suptitle(EOF_MODE)

cmap = cm.get_cmap("bwr")
ax = []
for i, casename in enumerate(casenames):

    exp_name = casename
    caseinfo = sim_casenames[exp_name]

    label = exp_name
    lc = caseinfo["lc"]
    ls = caseinfo["ls"]
        
    variances = data["CTL"][exp_name]["PCAs_ts_var"]

    for m in range(plotted_modes):
        _ax = fig.add_subplot(spec[i, m], **proj_kw)
        ax.append(_ax)
   
        eof = data["CTL"][exp_name]["PCAs"][m, beg_idx::sparsity, beg_idx::sparsity]

        if flip[m] is not None:
            eof *= flip[m][i]

        amp = np.nanstd(eof)
         
        mappable = _ax.contourf(lon_sparse, lat_sparse, eof,  np.linspace(-1, 1, 21) * amp * 1.5,  cmap=cm.get_cmap("bwr"), transform=data_proj, extend="both")
        if m == 0:
            _ax.set_title("CTL_%s " % (label, ))

        #if ax_idx[1] == 0:
        #    _ax.text(-0.02, 0.5, "EOF%d" % (m+1,), ha="right", va="center", transform=_ax.transAxes, rotation=90)
            
        _ax.text(0.02, 0.95, "%.1f%%" % (variances[m] / np.sum(variances) * 100,), ha="left", va="top", transform=_ax.transAxes, bbox={"facecolor":"white", "edgecolor":'none'})

for _ax in ax:

    #_ax.set_aspect('auto')

    _ax.coastlines()
    _ax.add_feature(cfeature.LAND, color="#cccccc")
    _ax.set_aspect('auto')

    gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.3, linestyle='-')
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([0, 60, 120, 180, -120, -60])
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {'size': 8, 'color': 'black', 'ha':'center'}
    gl.ylabel_style = {'size': 8, 'color': 'black', 'ha':'right'}
 
    if extent is not None:
        _ax.set_extent(extent, ccrs.PlateCarree())


    #_ax.set_ylim([-20, 20])
#    _ax.set_yticks([])#[-90, -60, -30, 0, 30, 60, 90])
#    _ax.set_yticklabels([])

fig.savefig("%s/fig05_EOF_map_%s.png" % (output_dir, EOF_MODE), dpi=300)
plt.show()
plt.close(fig)

