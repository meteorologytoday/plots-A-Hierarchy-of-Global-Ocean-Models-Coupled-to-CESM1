import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mplt
from matplotlib import cm
import matplotlib.path as mpath

import os
from matplotlib import rc

with open("./colordef.py") as infile:
    exec(infile.read())


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

with Dataset("data/supp/vice_target_file/forcing.vice.f09.paper2021_CTL_POP2.nc", "r") as f:
    vice_CTL_MM = f.variables["vice_target"][:]
    vice_CTL = np.mean(f.variables["vice_target"][:], axis=0)

with Dataset("data/supp/vice_target_file/forcing.vice.f09.paper2021_EXP_POP2.nc", "r") as f:
    vice_EXP_MM = f.variables["vice_target"][:]
    vice_EXP = np.mean(f.variables["vice_target"][:], axis=0)


plot_height = 1.0


data = {}

# calculating the ensemble difference
target = {"CTL": vice_CTL_MM, "EXP" : vice_EXP_MM}

projNH_kw = {
    'projection': ccrs.NorthPolarStereo(),
    'aspect' : 1,
}

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
widths  = [1, .05]
spec = fig.add_gridspec(nrows=1, ncols=2, width_ratios=widths, wspace=0.2) 

cmap = cm.get_cmap("Blues")
cmap.set_under("white")
clevs = np.linspace(0, 4, 11)

ax=[]
ax.append(fig.add_subplot(spec[0, 0], **projNH_kw))

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

NH_bnd = [ -180, 180,  50,  90 ]

ax[0].set_extent(NH_bnd, ccrs.PlateCarree())

for _ax in ax:
    _ax.add_feature(cfeature.LAND, color='#dddddd')
    _ax.add_feature(cfeature.BORDERS, linewidth=1.0)
    _ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    _ax.coastlines(zorder=0, color="#dddddd")
    _ax.gridlines(zorder=100)
    _ax.set_boundary(circle, transform=_ax.transAxes)

    _ax.outline_patch.set_linewidth(2)
    #_ax.outline_patch.set_linestyle(':')


for i, _ax in enumerate(ax):
#    _ax.contour(lon, lat, np.mean(vice_CTL_MM[1:4, :, :], axis=0), plot_height, transform=ccrs.PlateCarree(), colors=(CBFC["b"],), linestyles="-", zorder=11)
#    _ax.contour(lon, lat, np.mean(vice_CTL_MM[7:10, :, :], axis=0), plot_height, transform=ccrs.PlateCarree(), colors=(CBFC["b"],), linestyles="--", zorder=11)

#    _ax.contour(lon, lat, np.mean(vice_EXP_MM[1:4, :, :], axis=0), plot_height, transform=ccrs.PlateCarree(), colors=(CBFC["r"],), linestyles="-", zorder=11)
#    _ax.contour(lon, lat, np.mean(vice_EXP_MM[7:10, :, :], axis=0), plot_height, transform=ccrs.PlateCarree(), colors=(CBFC["r"],), linestyles="dashed", zorder=11)

    _ax.contourf(lon, lat, np.mean(vice_CTL_MM[:, :, :], axis=0), np.array([0, plot_height, 10]), transform=ccrs.PlateCarree(), zorder=11, colors=[(0,0,0,0), CBFC["b"]])
    _ax.contourf(lon, lat, np.mean(vice_EXP_MM[:, :, :], axis=0), np.array([0, plot_height, 10]), transform=ccrs.PlateCarree(), zorder=12, colors=[(0,0,0,0), CBFC["r"]])
#    _ax.contourf(lon, lat, np.mean(vice_EXP_MM[:, :, :], axis=0), np.array([0, plot_height, 10]), transform=ccrs.PlateCarree(), hatches=[None, '.'], zorder=11, colors="white", edgecolor="red")
    #_ax.contourf(lon, lat, np.mean(vice_CTL_MM[:, :, :], axis=0), np.array([0, plot_height, 10]), transform=ccrs.PlateCarree())
#    _ax.contour(lon, lat, np.mean(vice_CTL_MM[:, :, :], axis=0), [plot_height, transform=ccrs.PlateCarree(), colors=(CBFC["b"],), linestyles="solid", zorder=11)

#    _ax.contour(lon, lat, np.mean(vice_EXP_MM[:, :, :], axis=0), plot_height, transform=ccrs.PlateCarree(), colors=(CBFC["r"],), linestyles="solid", zorder=11)




#ax[0].annotate("CTL", xy=(85, 4), xytext=(45, 4.5), color=CBFC['b'], ha="center", va="center", arrowprops={'arrowstyle':'-|>', 'color': CBFC['b'], 'connectionstyle': connectionstyle})


    #_ax.text(0.01, 0.99, "(%s)" % "ab"[i], va="top", ha="left", transform=_ax.transAxes, size=30)


#mappable = ax[2].contour(lon, lat, vice_EXP_MM[2, :, :], , transform=ccrs.PlateCarree())

fig.savefig("figures/figS06_seaice_target_map_contour.png", dpi=600)

plt.show()
plt.close(fig)

