import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mplt
from matplotlib import cm
import matplotlib.path as mpath

import os


from matplotlib import rc

default_linewidth = 1.0;
default_ticksize = 10.0;

mplt.rcParams['lines.linewidth'] =   default_linewidth;
mplt.rcParams['axes.linewidth'] =    default_linewidth;
mplt.rcParams['xtick.major.size'] =  default_ticksize;
mplt.rcParams['xtick.major.width'] = default_linewidth;
mplt.rcParams['ytick.major.size'] =  default_ticksize;
mplt.rcParams['ytick.major.width'] = default_linewidth;

#rc('font', **{'family':'sans-serif', 'serif': 'Bitstream Vera Serif', 'sans-serif': 'MS Reference Sans Serif', 'size': 20.0});
rc('font', **{'size': 25.0});
rc('axes', **{'labelsize': 15.0});
rc('mathtext', **{'fontset':'stixsans'});
#rc(('xtick.major','ytick.major'), pad=20)

#import matplotlib.font_manager as fm;
#print("%s: %d"%(fm.FontProperties().get_name(),fm.FontProperties().get_weight()));

import matplotlib.pyplot as plt

import sys, argparse
from netCDF4 import Dataset
import numpy as np
from pprint import pprint

def ext(data):
    s = data.shape
    ndata = np.zeros((s[0], s[1]+1))
    ndata[:, 0:-1] = data
    ndata[:, -1] = data[:, 0]
    return ndata
 
def ext_axis(lon):
    return np.append(lon, 360) 

def area_mean(data, area):
    data.mask = False
    idx = np.isfinite(data)
    aa = area[idx]
    return sum(data[idx] * aa) / sum(aa)
 
domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"

with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:]
    lon  = f.variables["xc"][:]
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

model_set = ["CCSM4", "CESM1"][1]


vice = []

if model_set == "CCSM4":
    for i in range(6):
        filename = "data/seaice_ensemble/CCSM4_f09/sit_ens%02d.nc" % (i+1,)
        with Dataset(filename, "r") as f:
            vice.append(f.variables["sit"][:])

elif model_set == "CESM1":        
    for i in range(40):
        if i == 10:
            continue
        filename = "data/seaice_ensemble/CESM1_f09/sit_ens%02d.nc" % (i+1,)
        with Dataset(filename, "r") as f:
            vice.append(f.variables["sit"][:])
        

vice_ens = None
for i in range(len(vice)):
    if vice_ens is None:
        vice_ens = vice[i].copy()
    else:
        vice_ens += vice[i]

vice_ens /= len(vice)

print("We have %d ensemble members." % (len(vice)))





projNH_kw = {
    'projection': ccrs.NorthPolarStereo(),
    'aspect' : 1,
}

projSH_kw = {
    'projection': ccrs.SouthPolarStereo(),
    'aspect' : 1,
}

fig = plt.figure(constrained_layout=True, figsize=(5*6, 6*2))
spec = fig.add_gridspec(nrows=2, ncols=6, wspace=0.2) 

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

NH_bnd = [-180, 180, 50, 90]
SH_bnd = [-180, 180, -90, -50]


#ax[0].contour(lon, lat, vice, [0, 1.0, 100], transform=ccrs.PlateCarree())

fig.suptitle("%s (%d)" % (model_set, len(vice)))

ax=[]
m = 0
for row in range(2):
    for col in range(6):
        print("Plotting month : %02d" % (m+1,))
    
        _ax = fig.add_subplot(spec[row, col], **projNH_kw)
        ax.append(_ax)

        _ax.set_extent(NH_bnd, ccrs.PlateCarree())
        #_ax.add_feature(cfeature.BORDERS, linewidth=2.0)
        _ax.add_feature(cfeature.COASTLINE, linewidth=2.0, edgecolor="#888888")
        _ax.gridlines()
        _ax.set_boundary(circle, transform=_ax.transAxes)

        _ax.outline_patch.set_linewidth(2)

        _ax.set_title("%02d" % (m+1,))

        """
        for i in range(len(vice)):
            if model_set == "CCSM4" and i == 0:
                colors = ["red", "navy"]
                zorder = 99
                linewidth = 3
                linestyle = "solid"
            else:
                colors = ["tab:pink", "royalblue"]
                zorder = 50
                linewidth = 1
                linestyle = "solid"
            _ax.contour(lon, lat, vice[i][m, :, :], [0.5, 1.0,], transform=ccrs.PlateCarree(), colors=colors, linewidths=linewidth, linestyles=linestyle, zorder=zorder)

            
        _ax.contour(lon, lat, vice_ens[m, :, :], [0.5, 1.0,], transform=ccrs.PlateCarree(), colors=["darkred", "darkviolet"], linewidths=2, linestyles="--", zorder=100)
        """

        for i in range(len(vice)):
            if model_set == "CCSM4" and i == 0:
                colors = ["navy"]
                zorder = 99
                linewidth = 3
                linestyle = "solid"
            else:
                colors = ["royalblue"]
                zorder = 50
                linewidth = 1
                linestyle = "solid"
            _ax.contour(lon, lat, vice[i][m, :, :], [1.0,], transform=ccrs.PlateCarree(), colors=colors, linewidths=linewidth, linestyles=linestyle, zorder=zorder)

            
        _ax.contour(lon, lat, vice_ens[m, :, :], [1.0,], transform=ccrs.PlateCarree(), colors=["darkviolet"], linewidths=3, linestyles="--", zorder=100)

        m += 1

fig.savefig("graph/seaice_ensemble_%s.png" % (model_set,), dpi=600)

plt.show()
plt.close(fig)

