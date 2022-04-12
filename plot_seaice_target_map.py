import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mplt
from matplotlib import cm
import matplotlib.path as mpath

import os


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

with Dataset("data/vice_target_file/forcing.vice.f09.paper2021_CTL_POP2.nc", "r") as f:
    vice_CTL_MM = f.variables["vice_target"][:]
    vice_CTL = np.mean(f.variables["vice_target"][:], axis=0)

with Dataset("data/vice_target_file/forcing.vice.f09.paper2021_EXP_POP2.nc", "r") as f:
    vice_EXP_MM = f.variables["vice_target"][:]
    vice_EXP = np.mean(f.variables["vice_target"][:], axis=0)


vice_CTL[vice_CTL < .01] = -1
vice_EXP[vice_EXP < .01] = -1

data = {}

# calculating the ensemble difference
target = {"CTL": vice_CTL_MM, "EXP" : vice_EXP_MM}

projNH_kw = {
    'projection': ccrs.NorthPolarStereo(),
    'aspect' : 1,
}

projSH_kw = {
    'projection': ccrs.SouthPolarStereo(),
    'aspect' : 1,
}

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
widths  = [1, 1, .05]
spec = fig.add_gridspec(nrows=1, ncols=3, width_ratios=widths, wspace=0.2) 

cmap = cm.get_cmap("Blues")
cmap.set_under("white")
clevs = np.linspace(0, 4, 11)

ax=[]
ax.append(fig.add_subplot(spec[0, 0], **projNH_kw))
ax.append(fig.add_subplot(spec[0, 1], **projNH_kw))

#ax[0].set_title("(a) CTL", pad=20)
#ax[1].set_title("(b) EXP", pad=20)

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

NH_bnd = [-180, 180, 50, 90]
SH_bnd = [-180, 180, -90, -50]

ax[0].set_extent(NH_bnd, ccrs.PlateCarree())
ax[1].set_extent(NH_bnd, ccrs.PlateCarree())




for _ax in ax:
    _ax.add_feature(cfeature.BORDERS, linewidth=2.0)
    _ax.add_feature(cfeature.COASTLINE, linewidth=2.0)
    _ax.coastlines()
    _ax.gridlines()
    _ax.set_boundary(circle, transform=_ax.transAxes)

    _ax.outline_patch.set_linewidth(2)
    #_ax.outline_patch.set_linestyle(':')

mappable = ax[0].contourf(lon, lat, vice_CTL, clevs, cmap=cmap, extend="max", transform=ccrs.PlateCarree())
mappable = ax[1].contourf(lon, lat, vice_EXP, clevs, cmap=cmap, extend="max", transform=ccrs.PlateCarree())

cax = fig.add_subplot(spec[:, -1])
cb = fig.colorbar(mappable,  cax=cax, orientation="vertical")
cb.set_label("[$ \mathrm{m}^3 / \mathrm{m}^2 $]", fontsize=30)



fig.savefig("graph/seaice_target_map.png", dpi=600)

plt.show()
plt.close(fig)

