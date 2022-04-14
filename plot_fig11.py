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
output_dir = "figures"
plot_vars = ["PREC_TOTAL"]

sim_casenames = getSimcases(["SOM", "MLM", "EMOM", "OGCM"])
sim_var = getSimVars(plot_vars)

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

            filename = "data/hierarchy_statistics/%s/%s" % (data_dir, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s => %s" % (data_dir, varname))
                var_mean = "%s_AM" % varname
                var_std  = "%s_AASTD" % varname
                
                data[scenario][exp_name][var_mean] = f.variables[var_mean][:, 0, :, lon_idx]
                data[scenario][exp_name][var_std]  = f.variables[var_std][:, 0, :, lon_idx]

        
                

plot_infos = {

    "vice" : {
        "display"   : "SIT",
        "unit"      : "m",
        "var_mean"  : "vice_ZONAL_MM",
        "var_std"   : "vice_ZONAL_MASTD",
        "ylim_mean" : [-3, 3],
        "ylim_mean_diff" : [-3, 3],
        "ylim_std"  : [-1, 1],
        "factor"    : 1.0,
    },


    "TREFHT" : {
        "display"   : "Surface air temperature",
        "unit"      : "$^\\circ\\mathrm{C}$",
        "var_mean"  : "TREFHT_ZONAL_MM",
        "var_std"   : "TREFHT_ZONAL_MASTD",
        "ylim_mean_diff" : [-1, 10],
        "ylim_std"  : [-1, 1],
        "factor"    : 1.0,
        "yticks_mean" : [0, 5, 10],
    },

    "PREC_TOTAL" : {
        "display"   : "Precip",
        "unit"      : "mm / day",
        "var_mean"  : "PREC_TOTAL_ZONAL_MM",
        "var_std"   : "PREC_TOTAL_ZONAL_MASTD",
        "ylim_mean" :      [0, 8],
        "ylim_mean_diff" : [-0.4, 0.8],
        "ylim_mean_diff_ticks" : [-0.5, 0, 0.5],
        "ylim_std"  : [0, 10],
        "factor"       : 86400.0 * 1000.0,
    },

    "SST" : {
        "display"   : "SST",
        "unit"      : "$^\\circ\\mathrm{C}$",
        "var_mean"  : "SST_ZONAL_MM",
        "var_std"   : "SST_ZONAL_MASTD",
        "ylim_mean" : [-5, 30],
        "ylim_mean_diff" : [-0.1, 1.5],
        "ylim_std"  : [0, 10],
        "factor"        : 1.0,
        "yticks_mean" : [0, 0.5, 1.0, 1.5],
    },

    "TAUX" : {
        "display"   : "Surface Zonal Wind Stress",
        "unit"      : r"$ \mathrm{N} \; / \; \mathrm{m}^2 $",
        "var_mean"  : "TAUX_ZONAL_MM",
        "var_std"   : "TAUX_ZONAL_MASTD",
        "ylim_mean" : [-0.1, 0.1],
        "ylim_mean_diff" : [-0.01, 0.01],
        "ylim_std"  : [-0.05, 0.05],
        "factor"    : -1.0,
    },

    "TAUY" : {
        "display"   : "Surface Meridional Wind Stress",
        "unit"      : r"$ \mathrm{N} \; / \; \mathrm{m}^2 $",
        "var_mean"  : "TAUY_ZONAL_MM",
        "var_std"   : "TAUY_ZONAL_MASTD",
        "ylim_mean" : [-0.1, 0.1],
        "ylim_mean_diff" : [-0.01, 0.01],
        "ylim_std"  : [-0.05, 0.05],
        "factor"    : -1.0,
    },



    "h_ML" : {
        "display"   : "Mixed-layer Thickness",
        "unit"      : "m",
        "var_mean"  : "h_ML_ZONAL_MM",
        "var_std"   : "h_ML_ZONAL_MASTD",
        "ylim_mean" : [0, 10],
        "ylim_mean_diff" : [-10, 10],
        "ylim_std"  : [0, 10],
        "factor"        : 1.0,
    },

    "PSL" : {
        "display"   : "Sea-level Pressure",
        "unit"      : "hPa",
        "var_mean"  : "PSL_ZONAL_MM",
        "var_std"   : "PSL_ZONAL_MASTD",
        "ylim_mean" : [0, 10],
        "ylim_std"  : [0, 10],
        "factor"       : 1e-2,
    },

    "ICEFRAC" : {
        "display"   : "Sea-ice Concentration",
        "unit"      : "%",
        "var_mean"  : "ICEFRAC_ZONAL_MM",
        "var_std"   : "ICEFRAC_ZONAL_MASTD",
        "ylim_mean" : [0, 10],
        "ylim_std"  : [0, 10],
        "factor"       : 1e2,
    },

    "SWCF" : {
        "display"   : "Shortwave cloud forcing",
        "unit"      : r"$W / m^2$",
        "ylim_mean" : [0, 500],
        "ylim_std"  : [0, 100],
        "factor"    : 1.0,
        "ylim_mean_diff" : [-10, 10],
    },

    "LWCF" : {
        "display"   : "Longwave cloud forcing",
        "unit"      : r"$W / m^2$",
        "ylim_mean" : [0, 500],
        "ylim_std"  : [0, 100],
        "factor"    : 1.0,
        "ylim_mean_diff" : [-10, 10],
    },



    "FSNT" : {
        "display"   : "TOA",
        "unit"      : r"$W / m^2$",
        #"var_mean"  : "FSNT_ZONAL_MM",
        #"var_std"   : "FSNT_ZONAL_MASTD",
        "ylim_mean" : [0, 500],
        "ylim_std"  : [0, 100],
        "factor"    : 1.0,
        "ylim_mean_diff" : [-10, 10],
    },

    "FLNT" : {
        "display"   : "OLR",
        "unit"      : r"$W / m^2$",
        "var_mean"  : "FLNT_ZONAL_MM",
        "var_std"   : "FLNT_ZONAL_MASTD",
        "ylim_mean" : [0, 500],
        "ylim_std"  : [0, 100],
        "factor"    : 1.0,
        "ylim_mean_diff" : [-10, 10],
    },

    "NetRad" : {
        "display"   : "TOA Net Radiation",
        "unit"      : r"$\mathrm{W} / \mathrm{m}^2$",
        "var_mean"  : "NetRad",
        "var_std"   : "FLNT_ZONAL_MASTD",
        "ylim_mean" : [0, 500],
        "ylim_std"  : [0, 100],
        "factor"    : 1.0,
        "ylim_mean_diff" : [-1.0, 1.5],
    },

    "FLNT" : {
        "display"   : "OLR",
        "unit"      : r"$W / m^2$",
        "var_mean"  : "FLNT_ZONAL_MM",
        "var_std"   : "FLNT_ZONAL_MASTD",
        "ylim_mean" : [0, 500],
        "ylim_std"  : [0, 100],
        "factor"    : 1.0,
        "ylim_mean_diff" : [-10, 10],
    },

    "LHFLX" : {
        "display"   : "Latent heat flux",
        "unit"      : r"$W / m^2$",
        "var_mean"  : "LHFLX_ZONAL_MM",
        "var_std"   : "LHFLX_ZONAL_MASTD",
        "factor"    : 1.0,
        "ylim_mean_diff" : [-10, 10],
    },

}
           
try: 
    os.makedirs(output_dir)

except:
    pass


plot_vars = ["PREC_TOTAL",]

fig, ax = plt.subplots(len(plot_vars), 1, sharex=True, figsize=(8, 4), constrained_layout=True, squeeze=False)

ax = ax.flatten()

#fig.subplots_adjust(wspace=0.3)
thumbnail = "abcdefg"
for (i, varname) in enumerate(plot_vars):

    plot_info = plot_infos[varname]

    var_mean = "%s_AM" % varname
    var_std  = "%s_AASTD" % varname

    factor = plot_info["factor"]
    
    #ax[i].set_title("(%s) %s" % ( thumbnail[i], plot_info["display"]))
    #ax[i].set_title("Zonal mean of annual precipitation")

   
    for exp_name, caseinfo in sim_casenames.items():
      
        label = "RESP_%s" % exp_name
        lc = caseinfo["lc"]
        ls = caseinfo["ls"]

        _CTL_mean = np.mean(data["CTL"][exp_name][var_mean][:], axis=(0, 2,)) * factor
        _EXP_mean = np.mean(data["EXP"][exp_name][var_mean][:], axis=(0, 2,)) * factor
        _diff_mean = _EXP_mean - _CTL_mean

        if varname == "NetRad":
            _diff_mean *= np.cos(lat * np.pi / 180)


        if varname == "vice":
            ax[i].plot(lat, _CTL_mean, linestyle="dashed", color=lc, label=label)
            ax[i].plot(lat, _EXP_mean, linestyle=ls, color=lc, label=label)

        else:
            ax[i].plot(lat, _diff_mean, linestyle=ls, color=lc, label=label)

        if varname == "PREC_TOTAL" and exp_name == "OGCM":
            _ax = ax[i].twinx()
            _ax.plot(lat, _CTL_mean, "k--", linewidth=1, label="Precip of CTL_OGCM")
            #_ax.set_yticks([])
            _ax.set_ylabel("OGCM_CTL precip [ mm / day ]")
            _ax.set_ylim(np.array([-1, 2]) * 4)
            _ax.legend(fontsize=12, loc='lower left')



    ax[i].set_ylabel(r"%s response [ %s ]" % (plot_info["display"], plot_info["unit"]))
    ax[i].set_ylim(plot_info["ylim_mean_diff"])

    if 'yticks_mean' in plot_info:
        ax[i].set_yticks(plot_info["yticks_mean"])

    for _ax in ax.flatten():
        _ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        _ax.set_xticklabels([])
        _ax.grid(True)

    for _ax in ax[0:-1]:
        _ax.tick_params(axis='x', which='both',length=0)

    ax[-1].set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
    ax[-1].set_xlim([-90, 90])

ax[0].legend(ncol=1, loc='upper left', framealpha=1, fontsize=12, columnspacing=1.0, handletextpad=0.3)

fig.savefig("%s/fig11_diff_zmean_PREC.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)

