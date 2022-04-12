import cartopy.crs as ccrs
import matplotlib as mplt
from matplotlib import cm
from quick_tools import *

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



ref_casename = "OGCM"


domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"

atm_lev_file = "atm_lev.nc"
atm_ilev_file = "atm_ilev.nc"

output_dir = "graph"

sim_casenames = getSimcases(["OGCM", "EMOM", "MLM", "SOM"])
#sim_casenames = getSimcases(["SOM", "MLM", "EMOM",])
#sim_casenames = getSimcases(["MLM", "MLM_tau01", "OGCM"])
sim_var = getSimVars(["T", "U", "Z3"])

g0 = 9.8
Re = 6.371e6
Omega = 2 * np.pi / 86400.0
with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]

with Dataset(atm_lev_file, "r") as f:
    lev  = f.variables["lev"][:]

with Dataset(atm_ilev_file, "r") as f:
    ilev  = f.variables["ilev"][:]


def ddy(v, lat):
    lat_rad = np.pi / 180.0 * lat
    y = lat_rad * Re
    dy = y[1:] - y[:-1]

    dvdy = (v[:, 1:] - v[:, :-1]) / dy[None, :]

    return dvdy
    
def calGeoU(Z, lat):
    lat_mid = (lat[1:] + lat[:-1]) / 2
    lat_mid_rad = np.pi / 180.0 * lat_mid

    f_mid = 2 * Omega * np.sin(lat_mid_rad)

    dZdy = ddy(Z, lat)
    geoU = - dZdy / f_mid * g0


    return geoU, lat_mid
    



data = {}

load_scenarios = ["CTL", "EXP"]

for scenario in load_scenarios:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():

        data_dir = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/%s/%s" % (data_dir, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s => %s" % (data_dir, varname))
                var_mean = "%s_ZONAL_MM" % varname
                var_std  = "%s_ZONAL_MASTD" % varname
                
                data[scenario][exp_name][var_mean] = f.variables[var_mean][:, :, :]
                data[scenario][exp_name][var_std]  = f.variables[var_std][:, :, :]

            
try: 
    os.makedirs(output_dir)

except:
    pass


for m in [4]:#[0,2,4]:#range(5):

    rng=[
        [11, 0,  1],
        [ 2, 3,  4],
        [ 5, 6,  7],
        [ 8, 9, 10],
        slice(None),
    ][m]

    ext = [
        "DJF",
        "MAM",
        "JJA",
        "SON",
        "MEAN"
    ][m]

 

    heights = [1,] * len(sim_casenames.keys()) + [0.1,]
    widths = [1, ] * 2
    
    fig = plt.figure(figsize=(len(widths) * 5, 5 * (len(heights)-1)))
    spec = fig.add_gridspec(nrows=len(heights), ncols=len(widths), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.4) 

    #norm = mplt.colors.BoundaryNorm(boundaries=clevels_diff, ncolors=256)
            
    fig.suptitle("Left: $U$ response (shading), and $U_g$ response (contours)\nRight: $U$ response (shading), $U$ in CTL (red contours), and $dT/dy$ response (black contours)", size=12)

    idx=0

    for exp_name, caseinfo in sim_casenames.items():

        label = exp_name
        
        row_idx = idx#caseinfo["col_idx"]
        idx += 1

        ax = []

        for i in range(2):
            ax.append(fig.add_subplot(spec[row_idx, i]))

         
        
        raw_DAT = data["EXP"][exp_name]
        raw_REF = data["CTL"][exp_name]

        _DAT_U = np.mean(raw_DAT["U_ZONAL_MM"][rng, :, :], axis=(0,))
        _REF_U = np.mean(raw_REF["U_ZONAL_MM"][rng, :, :], axis=(0,))

        _DAT_Z3 = np.mean(raw_DAT["Z3_ZONAL_MM"][rng, :, :], axis=(0,))
        _REF_Z3 = np.mean(raw_REF["Z3_ZONAL_MM"][rng, :, :], axis=(0,))

        _DAT_T = np.mean(raw_DAT["T_ZONAL_MM"][rng, :, :], axis=(0,))
        _REF_T = np.mean(raw_REF["T_ZONAL_MM"][rng, :, :], axis=(0,))


        _diff_U  = _DAT_U  - _REF_U
        _diff_T  = _DAT_T  - _REF_T
        _diff_Z3 = _DAT_Z3 - _REF_Z3

        _diff_geoU, lat_mid = calGeoU(_diff_Z3, lat)
        _diff_dTdy = ddy(_diff_T, lat)


        levels_U_diff = np.linspace(-1, 1,  11)
        mappable_diff = ax[0].contourf(lat, lev, _diff_U, levels_U_diff, cmap=cm.get_cmap("bwr"), extend="both")
        cs = ax[0].contour(lat_mid, lev, _diff_geoU, levels_U_diff, colors='k', linewidths=1)
        labels = plt.clabel(cs, fmt="%.1f", fontsize=10)


        mappable_diff2 = ax[1].contourf(lat, lev, _diff_U, levels_U_diff, cmap=cm.get_cmap("bwr"), extend="both")
        cs3 = ax[1].contour(lat, lev, _REF_U, 21, colors='r', linewidths=1)
        levels_dTdy_diff = np.linspace(-1, 1,  11) 
        cs2 = ax[1].contour(lat_mid, lev, _diff_dTdy / 1e-7, levels_dTdy_diff, colors='k', linewidths=1)
        labels2 = plt.clabel(cs2, fmt="%.1f", fontsize=10)
       
        print("Max dTdy", np.amax(_diff_dTdy)) 
        print("Min dTdy", np.amin(_diff_dTdy)) 
        for _ax in ax:
            _ax.set_title("RESP_%s" % (label,), size=15)
            _ax.grid(True) 
             
            _ax.set_xlim([-90, 90])
            _ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            _ax.set_xticklabels([])
            _ax.set_yticks([1000, 750, 500, 250, 100])
            _ax.set_yticklabels([""]*5)
            _ax.invert_yaxis()
            _ax.set_yticklabels(["1000", "750", "500", "250", "100"], fontsize=10)
            _ax.set_ylabel("[ hPa ]", fontsize=10)

            if row_idx == len(heights) - 2:
                _ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"], fontsize=10)
                #_ax.set_xticklabels(["90S", "", "", "EQ", "", "", "90N"])
        


       
 
        cax = fig.add_subplot(spec[-1, 0])
        cb_diff = fig.colorbar(mappable_diff, cax=cax, orientation="horizontal", ticks=levels_U_diff)
        cb_diff.ax.tick_params(labelsize=12)
        cb_diff.set_label("[ $ \\mathrm{m} / \\mathrm{s} $ ]")

        cax2 = fig.add_subplot(spec[-1, 1])
        cb_diff2 = fig.colorbar(mappable_diff2, cax=cax2, orientation="horizontal", ticks=levels_dTdy_diff)
        cb_diff2.ax.tick_params(labelsize=12)
        cb_diff2.set_label("[ $ \\mathrm{K} / \\mathrm{m} $ ]")


    plt.show()
    fig.savefig("graph/Thermal_wind.png", dpi=600)
    plt.close(fig)

