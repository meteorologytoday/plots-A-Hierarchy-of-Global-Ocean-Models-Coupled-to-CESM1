import matplotlib as mplt
from matplotlib import cm

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

def mavg(y, w):

    N = len(y)
    yy = np.zeros((N,))
    if w == 0:
        yy[:] = y

    else: 
    
        window = w * 2 + 1
        for i in range(N):
            if i < w:
                rng = slice(0, i+w+1)
                yy[i] = np.mean(y[rng])

            elif i > (N - w - 1):
                rng = slice(i-w, N)
                yy[i] = np.mean(y[rng])

            else:
                rng = slice(i-w,i+w+1)
                yy[i] = np.mean(y[rng])
        
    return yy



moc_ctl_t_rng = slice((1-1)*12,  (100-1)*12+12)
moc_exp_t_rng = slice((80-1)*12, (180-1)*12+12)


data = {}

scenarios = ["CTL", "EXP"]

print("Loading AMOC...")
for scenario in scenarios:

    d = {}
    with Dataset("data/AMOC/processed-MOC_%s.nc" % (scenario,), "r") as f:
        d["AMOC_max"]     = f.variables["AMOC_max"][:]
        d["AMOC_max_lat"] = f.variables["AMOC_max_lat"][:]
        d["AMOC_max_z"]   = f.variables["AMOC_max_z"][:]

    d["t"] = range(len(d["AMOC_max"]))
    data[scenario] = d


print("Loading streamfunction...")
# loading streamfunction
moc_lat = None
moc_z   = None
moc_data = {}
for scenario in scenarios:

    print(scenario)
    moc_data[scenario] = {}

    filename = "data/AMOC/MOC_%s.nc" % (scenario,)
    
    with Dataset(filename, "r") as f:
        moc_data[scenario]["MOC"] = f.variables["MOC"][:, :, 0, :, :]

        if moc_lat is None:
            moc_lat = f.variables["lat_aux_grid"][:]
            moc_z   = f.variables["moc_z"][:] / 100


print("Taking averages...")

MOC_CTL = np.mean(moc_data["CTL"]["MOC"][moc_ctl_t_rng, :, :, :], axis=0)
MOC_EXP = np.mean(moc_data["EXP"]["MOC"][moc_exp_t_rng, :, :, :], axis=0)

# Take Atlantic ocean portion out
MOC_DIFF = MOC_EXP - MOC_CTL
#MOC_CTL = MOC_CTL[1, :, :] #- MOC_CTL[1, :, :]
#MOC_EXP = MOC_EXP[1, :, :] #- MOC_EXP[1, :, :]


print("Plotting...")

fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, gridspec_kw={'width_ratios': [2.5, 1]})
#fig, ax = plt.subplots(2, 1, figsize=(6, 4), constrained_layout=True)

for s in scenarios:
    if s == "CTL":
        plot_t_rng = slice(0, 100,None)
    else:
        plot_t_rng = slice(0,None,None)
       

    lc = {
        "CTL" : CBFC['b'],
        "EXP" : CBFC["r"],
    }[s]
 
    ax[0].plot(data[s]["t"][plot_t_rng], data[s]["AMOC_max"][plot_t_rng], label="%s_OGCM" % s[0:3], color=lc)

ax[0].legend()
ax[0].set_xlabel("Time [ year ]")
ax[0].set_ylabel(r"[ Sv ]")
ax[0].set_ylim([20, 28])
ax[0].set_xlim([0, 180])
ax[0].set_title("(a) Timeseries of AMOC strength")


ax[1].plot([0, 0], [moc_z[0], moc_z[-1]], color="gray", linestyle="-")
CS = ax[1].contour(moc_lat, moc_z, MOC_DIFF[1, :, :], np.linspace(-8,0,9), colors="k", linewidths=2)

ax[1].clabel(CS, CS.levels, fmt="%d")



ax[1].set_ylim([0, 4000])
ax[1].set_xlim([30, 60])
#ax.set_xlabel("Latitude")
ax[1].set_ylabel("Depth [ m ]")
ax[1].set_xticks([30, 40, 50, 60])
#ax[1].set_xticklabels(["60S", "30S", "EQ", "30N", "60N"])

#ax[1].xaxis.tick_top()
ax[1].invert_yaxis()
ax[1].set_title("(b) Response of AMOC streamfunction (years 51-100)")
#ax[0].text(0.1, 0.9, "(a)", va="top", ha="left", transform=ax[0].transAxes)
#ax[1].text(0.1, 0.9, "(b)", va="top", ha="left", transform=ax[1].transAxes)


fig.savefig("graph/AMOC_timeseries_and_psi.png", dpi=600)
plt.show()
