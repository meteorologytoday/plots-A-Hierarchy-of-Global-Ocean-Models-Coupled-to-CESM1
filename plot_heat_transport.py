import cartopy.crs as ccrs
import matplotlib as mplt
from matplotlib import cm
import matplotlib.transforms as transforms
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
rc('font', **{'size': 12.0});
rc('axes', **{'labelsize': 10.0});
rc('mathtext', **{'fontset':'stixsans'});
#rc(('xtick.major','ytick.major'), pad=20)

#import matplotlib.font_manager as fm;
#print("%s: %d"%(fm.FontProperties().get_name(),fm.FontProperties().get_weight()));

import matplotlib.pyplot as plt

import sys, argparse
from netCDF4 import Dataset
import numpy as np
from pprint import pprint

sim_casenames = getSimcases(["SOM", "MLM", "EMOM", "OGCM_351-450"])
sim_var = getSimVars(["AHT", "OHT",])

def divergence(flux, lat):

    _lat_V = np.pi / 180.0 * lat
    _lat_T = ( _lat_V[:-1] + _lat_V[1:] ) / 2.0
    _dlat_T = _lat_V[1:] - _lat_V[:-1]
    _cos_V = np.cos( _lat_V )
    _cos_T = np.cos( _lat_T )
   
    wflux = flux #* _cos_V

    R = 6371e3
    div = ( wflux[1:] - wflux[:-1] ) / ((_cos_T * 2 * np.pi * R) * (_dlat_T * R))
    return div , _lat_T * 180.0 / np.pi


domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "graph"
        
OGCM_list = ["OGCM",]

with Dataset("CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc", "r") as f:
    lat_atm = f.variables["yc"][0, :]

lat = np.linspace(-90, 90, 181)

data = {}

for scenario in ["CTL", "EXP"]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():
            
        casename = caseinfo[scenario]
        d = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/%s/%s" % (casename, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s => %s" % (casename, varname))
                d[varname] = f.variables[varname][:]

        d["AHT"] = d["AHT"].mean(axis=0)
        d["OHT"] = d["OHT"].mean(axis=0)
        d["SUM"] = d["AHT"] + d["OHT"]

        data[scenario][exp_name] = d

"""
for sim_name, detail in OGCM_info.items():
    
    for scenario in ["CTL", "EXP"]:

        # Load total heat transport
        with Dataset("data/%s/POP2_OHT/%s" % (time_folder, detail[scenario]), "r") as f:

            d = data[scenario][sim_name]

            d["OHT_TOTAL"] = f.variables["N_HEAT"][0,tran_reg,tran_com,:]
            d["OHT_INDPAC"] = (f.variables["N_HEAT"][0, 0, 0,:] - f.variables["N_HEAT"][0, 1, 1, :])
            d["OHT_ATL"] = (f.variables["N_HEAT"][0, 1, 1, :])

            # interpolation
            for varname in ["OHT_TOTAL", "OHT_INDPAC", "OHT_ATL"]:
                d[varname] = np.interp(lat, lat_POP2, d[varname]) * 1e15

            d["SUM"] = d["AHT"] + d["OHT_TOTAL"]
"""


lat_plot_take_sin = True
#plot_cases = ["OGCM_ext200", "OGCM", "EMOM", "EOM", "SOM"]
plot_cases = ["OGCM_351-450", "EMOM", "MLM", "SOM"]

fig, ax = plt.subplots(len(plot_cases), 1, figsize=(5, 15), sharex=True, sharey=False, constrained_layout=True)

factor = 1e-15
lat_plot = lat    

tick_lat = np.array([-90, -60, -30, 0, 30, 60, 90])
tick_lat_labels = ["90S", "60S", "30S", "EQ", "30N", "60N", "90N"]

lat_lim = np.array([-90, 90])

tick_y = np.array([-0.2, -0.1, 0, 0.1, 0.2])
ylim = np.array([-0.2, 0.2])

if lat_plot_take_sin is True:
    lat_plot = np.sin(lat_plot * np.pi / 180.0)
    tick_lat = np.sin(tick_lat * np.pi / 180.0)
    lat_lim  = np.sin(lat_lim  * np.pi / 180.0)


for i, exp_name in enumerate(plot_cases):
    
    caseinfo = sim_casenames[exp_name] 

    print("Now doing: %d, %s" % (i, exp_name,))

    _ax = ax[i]

    label = exp_name
    lc = caseinfo["lc"]
    ls = caseinfo["ls"]

    _CTL_AHT = data["CTL"][exp_name]["AHT"] * factor
    _CTL_OHT = data["CTL"][exp_name]["OHT"] * factor

    _EXP_AHT = data["EXP"][exp_name]["AHT"] * factor
    _EXP_OHT = data["EXP"][exp_name]["OHT"] * factor

    _DIF_AHT = _EXP_AHT - _CTL_AHT
    _DIF_OHT = _EXP_OHT - _CTL_OHT

    _ax.plot(lat_plot, _DIF_AHT+_DIF_OHT, linestyle="solid",  color="black", label=r"$\Delta$PHT", zorder=14)
    _ax.plot(lat_plot, _DIF_AHT, linestyle="solid", color=CBFC['r'], label=r"$\Delta$AHT", zorder=13)
    _ax.plot(lat_plot, _DIF_OHT, linestyle="solid",  color=CBFC['b'], label=r"$\Delta$OHT", zorder=12)
    
   

    _ax.set_ylabel("[PW]")
#    trans = transforms.blended_transform_factory(_ax.transData, _ax.transAxes)
    _ax.text(np.sin(-43 * np.pi/180.0), -0.15, label, ha='center', va='center', transform=_ax.transData)

    if i == 0:
        _ax.legend(ncol=5, columnspacing=0.4, handletextpad=0.3, fontsize=8, loc='upper center', handlelength=2)
 
"""        
    if exp_name == 'OGCM':

        _EXP_ATL = data["EXP"][exp_name]["OHT_ATL"] * factor
        _CTL_ATL = data["CTL"][exp_name]["OHT_ATL"] * factor
        _diff_ATL = _EXP_ATL - _CTL_ATL
        _ax.plot(lat_plot, _diff_ATL, linestyle=":", color=CBFC['b'], label=r"$\Delta$OHT$_{ATL}$", zorder=11)

        _EXP_INDPAC = data["EXP"][exp_name]["OHT_INDPAC"] * factor
        _CTL_INDPAC = data["CTL"][exp_name]["OHT_INDPAC"] * factor
        _diff_INDPAC = _EXP_INDPAC - _CTL_INDPAC
        _ax.plot(lat_plot, _diff_INDPAC, dashes=[4,2], color=CBFC['b'], label=r"$\Delta$OHT$_{INDPAC}$", zorder=11)
"""


#ax[0].set_title("EXP - CTL")

for _ax in ax.flatten():

    _ax.set_xticks(tick_lat)
    _ax.set_xticklabels([""] * len(tick_lat_labels), size=10)
    _ax.set_xlim(lat_lim)

    _ax.grid(True)
    
    _ax.set_yticks(tick_y)


ax[-1].set_xticklabels(tick_lat_labels)

for _ax in ax.flatten():
    #_ax.set_ylim([-8, 8])
    #_ax.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])

    _ax.set_ylim(ylim)



#for _ax in ax[:, 1].flatten():
#    _ax.set_yticklabels([""] * len(_ax.get_yticks()))



plt.show()
fig.savefig("graph/heat_transport_analysis.png", dpi=600)

"""

fig, ax = plt.subplots(2, 1, sharex=True)

for exp_name, caseinfo in sim_casenames.items():
 
    label = exp_name
    lc = caseinfo["lc"]
    ls = caseinfo["ls"]

    _EXP = data["EXP"][exp_name]["AHT"]
    _CTL = data["CTL"][exp_name]["AHT"]

    _CTL_divAHT, lat_T = divergence(_CTL, lat)
    _EXP_divAHT, _ = divergence(_EXP, lat)

    _diff_divAHT = _EXP_divAHT - _CTL_divAHT

    ax[0].plot(lat_T, _CTL_divAHT, linestyle=ls, color=lc, label=label)
    ax[1].plot(lat_T, _diff_divAHT, linestyle=ls, color=lc, label=label)


ax[0].set_xlim([-10, 10])
ax[0].legend()
plt.show()

"""
