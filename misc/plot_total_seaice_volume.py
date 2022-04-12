import cartopy.crs as ccrs
import matplotlib as mplt
from matplotlib import cm
from quick_tools import *


import os

#mplt.use('Agg')

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


def monthly_mean(data):
    return np.mean(np.reshape(data, (12, -1), order='F'), axis=1)


domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "graph"

sim_casenames = getSimcases(["SOM", "MLM", "EMOM", "OGCM"])
sim_var = getSimVars([
    "ice_volume_NH",
    "ice_volume_SH",
    "ice_area_NH",
    "ice_area_SH",
])


with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]

data = {}

for scenario in ["CTL", "EXP"]:
#for scenario in ["CTL", ]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():
            
        data_dir = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/%s/%s" % (data_dir, filename, )
            
            with Dataset(filename, "r") as f:
                print("[%s] %s => %s" % (filename, data_dir, varname))
                data[scenario][exp_name][varname] = f.variables[varname][:]



plot_infos = {

    "ICE_SH" : {
        "display"   : "Total sea-ice volume in the southern hemisphere",
        "unit"      : r"$ \times 10^3 \, \mathrm{km}^3 $",
        "var"       : "ice_volume_SH",
        "ylim_mean" : [-0.1, 0.1],
        "factor"    : 1e-12,
        "CTL_rng"       : [10, 30],
        "EXP_rng"       : [10, 30],
    },


    "ICE_NH" : {
        "display"   : "Total sea-ice volume the in northern hemisphere",
        "unit"      : r"$ \times 10^3 \, \mathrm{km}^3 $",
        "var"       : "ice_volume_NH",
        "ylim_mean" : [-0.1, 0.1],
        "factor"    : 1e-12,
        "CTL_rng"       : [0, 45],
        "EXP_rng"       : [0, 45],
    },

    "ICE" : {
        "display"   : "Total sea-ice volume",
        "unit"      : r"$ \times 10^3 \, \mathrm{km}^3 $",
        "var"       : "ice_volume_GLB",
        "ylim_mean" : [-0.1, 0.1],
        "factor"    : 1e-12,
        "CTL_rng"       : [10, 65],
        "EXP_rng"       : [10, 65],
    },



}
           
factor_volume = 1e-12
factor_area   = 1e-12

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 8), constrained_layout=False)

fig.subplots_adjust(
    bottom=0.2,
)

t = list(range(12))

# plot target
#ax.plot(t, monthly_mean(targets["CTL"]["ice_volume_NH"]) * factor, color='gray', ls=":", lw="3", label="Target")

#ax.plot(t, monthly_mean(targets["EXP4"]["ice_volume_NH"]) * factor, color='gray', ls=":", lw="3", label="Target")

for exp_name, caseinfo in sim_casenames.items():
  
    label = exp_name

    lc_CTL = caseinfo["lc"]
    lc_EXP = caseinfo["lc"]

#    _CTL_NH = monthly_mean(data["CTL"][exp_name]["ice_volume_SH"]) * factor
#    _EXP_NH = monthly_mean(data["EXP"][exp_name]["ice_volume_SH"]) * factor

    for i, SIDE in enumerate(["NH", "SH"]):

        var_volume = "ice_volume_%s" % (SIDE,)
        var_area   = "ice_area_%s" % (SIDE,)

        _CTL_volume = monthly_mean(data["CTL"][exp_name][var_volume]) * factor_volume
        _EXP_volume = monthly_mean(data["EXP"][exp_name][var_volume]) * factor_volume
        ax[0, i].plot(t, _CTL_volume, linestyle="-", color=lc_CTL, label="CTL_%s" % label)
        ax[0, i].plot(t, _EXP_volume, linestyle="--", color=lc_EXP, label="SIL_%s" % label)#, label=label)


        _CTL_area = monthly_mean(data["CTL"][exp_name][var_area]) * factor_area
        _EXP_area = monthly_mean(data["EXP"][exp_name][var_area]) * factor_area
        ax[1, i].plot(t, _CTL_area, linestyle="-", color=lc_CTL, label=label)
        ax[1, i].plot(t, _EXP_area, linestyle="--", color=lc_EXP)#, label=label)


        #print("CTL Area %s - %s : %.2e" % (SIDE, exp_name, _CTL_area.mean()))
        #print("EXP Area %s - %s : %.2e" % (SIDE, exp_name, _EXP_area.mean()))
        print("CTL Vol  %s - %s : %.2e" % (SIDE, exp_name, _CTL_volume.mean()))
        print("EXP Vol  %s - %s : %.2e" % (SIDE, exp_name, _EXP_volume.mean()))




leg = fig.legend(handles=ax[0, 0].get_lines(), bbox_to_anchor=(0.5, 0.10), ncol=4, loc='upper center', framealpha=0.0)

#for _item in leg.legendHandles:
#    _item.set_color('black')


for _ax in ax.flatten():
    _ax.set_xticks(range(12))
    _ax.set_xticklabels(["" for _ in range(12)])
    _ax.grid(True)

for _ax in ax[-1, :]:
    ax[1, 0].set_xticklabels(["%d" % (i+1,) for i in range(12)])
#ax.set_yticks(np.linspace(0, 40, 5))

#ax[0, 0].text(8, 38, "CTL",  color=CBFC['b'], transform=ax[0,0].transData, va="bottom", ha="center")
#ax[0, 0].text(8, 2.5, "EXP", color=CBFC['r'], transform=ax[0,0].transData, va="bottom", ha="center")

#ax[1, 0].text(8, 10, "CTL",  color=CBFC['b'], transform=ax[1,0].transData, va="bottom", ha="center")
#ax[1, 0].text(8, 2.5, "EXP", color=CBFC['r'], transform=ax[1,0].transData, va="bottom", ha="center")


#ax.set_ylim([0, 45])
ax[0, 0].set_ylabel(r"[ $ \times 10^3 \, \mathrm{km}^3$ ]")
ax[1, 0].set_ylabel(r"[ $ \times 10^6 \, \mathrm{km}^2$ ]")

ax[0, 0].set_title("Arctic sea-ice volume")
ax[1, 0].set_title("Arctic sea-ice area")
ax[0, 1].set_title("Antarctic sea-ice volume")
ax[1, 1].set_title("Antarctic sea-ice area")


fig.savefig("graph/seaice_total_volume.png", dpi=600)
plt.show()
plt.close(fig)

