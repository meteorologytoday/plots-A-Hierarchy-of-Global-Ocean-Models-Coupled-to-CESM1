import cartopy.crs as ccrs
import matplotlib as mplt
from matplotlib import cm
from quick_tools import *

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
rc('font', **{'size': 15.0});
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

time_folder = "consistent_50years"
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
 
domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "graph"

plot_cases = ["EMOM"]
sim_casenames = getSimcases(plot_cases)
sim_var = getSimVars(["TAUX", "TAUY"])

with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]


lon_idx = np.logical_and(lon > 150, lon < 250) 
lon_idx = lon > -1 

data = {}

for scenario in ["CTL", "EXP"]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():
            
        data_dir = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/%s/%s" % (data_dir, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s => %s" % (data_dir, varname))
                var_mean = "%s_AM" % varname
                var_std  = "%s_AASTD" % varname
                
                data[scenario][exp_name][var_mean] = f.variables[var_mean][:, 0, :, lon_idx]
                data[scenario][exp_name][var_std]  = f.variables[var_std][:, 0, :, lon_idx]

        
                


plot_infos = {

    "TAUX" : {
        "display"   : r"$\tau_x$",
        "unit"      : r"$ \mathrm{N} \; / \; \mathrm{m}^2 $",
        "var_mean"  : "TAUX_ZONAL_MM",
        "var_std"   : "TAUX_ZONAL_MASTD",
        "ylim_mean" : [-0.1, 0.1],
        "ylim_mean_diff" : [-0.01, 0.01],
        "ylim_std"  : [-0.05, 0.05],
        "factor"    : -1e3,
    },

    "TAUY" : {
        "display"   : r"$\tau_y$",
        "unit"      : r"$ \mathrm{N} \; / \; \mathrm{m}^2 $",
        "var_mean"  : "TAUY_ZONAL_MM",
        "var_std"   : "TAUY_ZONAL_MASTD",
        "ylim_mean" : [-0.1, 0.1],
        "ylim_mean_diff" : [-0.01, 0.01],
        "ylim_std"  : [-0.05, 0.05],
        "factor"    : -1e3,
    },
}
           
try: 
    os.makedirs(output_dir)

except:
    pass



fig, ax = plt.subplots(1, len(plot_cases), sharex=True, figsize=(8, 4), constrained_layout=True, squeeze=False)

ax = ax.flatten()

#fig.subplots_adjust(wspace=0.3)
thumbnail = "abcdefg"

i=0
for exp_name, caseinfo in sim_casenames.items():

    for varname in ["TAUX", "TAUY"]:

        plot_info = plot_infos[varname]

        var_mean = "%s_AM" % varname
        var_std  = "%s_AASTD" % varname

        factor = plot_info["factor"]
        
        label = "%s of RESP_%s" % ( plot_info["display"], exp_name)
        
        ls = {
            "TAUX" : "solid",
            "TAUY" : "dashed",
        }[varname]

        _CTL_mean = np.mean(data["CTL"][exp_name][var_mean][:], axis=(0, 2,)) * factor
        _EXP_mean = np.mean(data["EXP"][exp_name][var_mean][:], axis=(0, 2,)) * factor
        _diff_mean = _EXP_mean - _CTL_mean

        ax[i].plot(lat, _diff_mean, linestyle=ls, color="black", label=label)



ax[0].set_ylabel(r"[ $\times 10^{-3} \, \mathrm{N}\, /\, \mathrm{m}^2 $ ]")
ax[0].set_ylim([-8, 2.5])

for _ax in ax.flatten():
    _ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    _ax.set_xticklabels([])
    _ax.grid(True)

    for _ax in ax[0:-1]:
        _ax.tick_params(axis='x', which='both',length=0)

    ax[-1].set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
    ax[-1].set_xlim([-90, 90])

    i += 1
#fig.subplots_adjust(bottom=0.2)    
#fig.legend(handles=ax[0].get_lines(), bbox_to_anchor=(0.5, 0.15), ncol=3, loc='upper center', framealpha=0.0)
ax[0].legend(ncol=1, loc='lower left', framealpha=1, fontsize=15, columnspacing=1.0, handletextpad=0.3)

#fig.savefig("%s/compare_exp_minus_SST_precip.png" % (output_dir,), dpi=200)
fig.savefig("graph/diff_zmean_TAU.png", dpi=600)
plt.show()
plt.close(fig)

