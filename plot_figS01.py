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


nyears = 50

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


def zm(data, lat_bnd, llat, valid_idx):
    o = np.zeros(len(lat_bnd)-1)
    for i in range(len(o)):
        idx = np.logical_and( np.logical_and(llat < lat_bnd[i+1], llat >= lat_bnd[i]), valid_idx)
        o[i] = data[idx].mean()

    return o


 
domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "figures"

Re = 6371e3
with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]
    dx = 2 * np.pi * np.cos(f.variables["yc"][:] * np.pi / 180.0) * Re / len(lon)
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

lnd_mask_idx = (mask == 1.0)

data = {}

print("Loading data")
        
with Dataset("data/supp/importance_of_KH/atm_21-30.nc", "r") as f:
    data["precip"] = f.variables["PRECC"][0, :, :] + f.variables["PRECL"][0, :, :]
    data["TAUY"] = f.variables["TAUY"][0, :, :]

with Dataset("data/supp/importance_of_KH/ocn_21-30.nc", "r") as f:
    data["TEMP"] = f.variables["TEMP"][0, :, :, :]
    data["VVEL"] = f.variables["VVEL"][0, :, :, :]
    data["WVEL"] = f.variables["WVEL"][0, :, :, :]
    lat_sT = f.variables["lat_sT"][:]
    lon_sT = f.variables["lon_sT"][:]

print("Data loaded.")

llon, llat = np.meshgrid(lon, lat)

lat_rng = [-20,  20]
lon_rng = [190, 210]

lat_bnd = np.linspace(lat_rng[0], lat_rng[1], 41)
lat_c = 0.5 * (lat_bnd[1:] + lat_bnd[:-1])

idx_atm = np.logical_and( np.logical_and(llon < lon_rng[1], llon > lon_rng[0]), np.logical_and(llat < lat_rng[1], llat > lat_rng[0])) 
idx_ocn = np.logical_and( np.logical_and(lon_sT < lon_rng[1], lon_sT > lon_rng[0]), np.logical_and(lat_sT < lat_rng[1], lat_sT > lat_rng[0])) 

try: 
    os.makedirs(output_dir)

except:
    pass

print("Creating figure object...")
fig, ax = plt.subplots(3, 1, figsize=(6, 8), constrained_layout=True, sharex=False)
print("Created.")
            


mappable = ax[0].contourf(lon, lat, data["precip"] * 86400.0 * 1000, np.linspace(0, 10, 11), cmap=cm.get_cmap("GnBu"), extend="both")
ax[0].contour(lon, lat, mask, [0.5], colors='black', linewidths=1)
ax[0].add_patch(patches.Rectangle((lon_rng[0], lat_rng[0]), lon_rng[1] - lon_rng[0], lat_rng[1] - lat_rng[0], linewidth=1, edgecolor='r', facecolor='none'))


#ax[1].scatter(llat[idx_atm], data["precip"][idx_atm] * 86400.0 * 1000, s=5)
ax[1].plot(lat_c, zm(data["precip"], lat_bnd, llat, idx_atm) * 1e3 * 86400.0, 'k-')




tax1 = ax[1].twinx()
tax1.plot(lat_c, - zm(data["TAUY"], lat_bnd, llat, idx_atm), 'r--')

# SST
ax[2].plot(lat_c, zm(data["TEMP"][0:1, :, :], lat_bnd, lat_sT, idx_ocn), 'k-')
tax2 = ax[2].twinx()
tax2.plot(lat_c, zm(data["WVEL"][5:6, :, :] * 86400.0, lat_bnd, lat_sT, idx_ocn), 'r--')


ax[1].set_ylabel(r"Precip [$\mathrm{mm}\,/\,\mathrm{day}$]")
ax[2].set_ylabel(r"SST [${}^\circ\mathrm{C}$]")

# Right spine 
tax1.tick_params(color='red', labelcolor='red')
tax1.spines["right"].set_edgecolor("red")
tax1.set_ylim([-0.05, 0.05])
tax1.set_ylabel(r"$\tau_y$ [$\mathrm{N}\,/\,\mathrm{m}^2$]", color="red")

tax2.tick_params(color='red', labelcolor='red')
tax2.spines["right"].set_edgecolor("red")
tax2.set_ylim([-1.5, 1.5])
tax2.set_ylabel(r"$w_{50\mathrm{m}}$ [m / day]", color="red")

cb = fig.colorbar(mappable, ax=ax[0], ticks=np.linspace(0, 10, 5), orientation="vertical")
cb.set_label("%s %s" % ("Precip", "[mm / day]"))       

ax[0].set_ylim([-45, 45])
ax[0].set_yticks([-45, -30, -15, 0, 15, 30, 45])
ax[0].set_xlim([100, 300])

for _ax in ax[1:]:
    _ax.set_xlim([-20, 20])
    _ax.set_xticks([-20, -10, 0, 10, 20])


for (i, _ax) in enumerate(ax):
    _ax.text(0.03, 0.95, "(%s)" % ( "abcdef"[i], ), ha="left", va="top", transform=_ax.transAxes)


fig.savefig("%s/figS01_importance_of_KH.png" % (output_dir, ), dpi=600)
plt.show()
plt.close(fig)

