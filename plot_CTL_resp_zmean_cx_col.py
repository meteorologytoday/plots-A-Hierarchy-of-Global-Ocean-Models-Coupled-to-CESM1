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


plot_type = ["diff_OGCM", "diff_EXP"][1]

ref_casename = "OGCM"



nyears = 50
dof = 2 *nyears - 2 # degree of freedom

domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"

atm_lev_file = "atm_lev.nc"
atm_ilev_file = "atm_ilev.nc"

output_dir = "graph"

sim_casenames = getSimcases(["OGCM", "EMOM", "MLM", "SOM"])
#sim_casenames = getSimcases(["MLM", "MLM_tau01", "OGCM"])
sim_var = getSimVars(["T", "U"])

Re = 6371e3
with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]

with Dataset(atm_lev_file, "r") as f:
    lev  = f.variables["lev"][:]

with Dataset(atm_ilev_file, "r") as f:
    ilev  = f.variables["ilev"][:]

data = {}

if plot_type == "diff_OGCM":
    load_scenarios = ["CTL"]

elif plot_type == "diff_EXP":
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


plot_infos = {

    "PSI" : {
        "display"      : "streamfunction",
        "unit"         : r"\times 10^{9} \, \mathrm{kg} \, / \, \mathrm{s}",
        "var"          : "psi_ZONAL_MM",
        "var_std"      : "psi_ZONAL_MASTD",
        "cmap_mean"    : "bwr",
        "clevels_mean" : np.linspace(-100, 100, 11),
        "clevels_diff" : np.linspace(-10, 10,  11),
        "factor"       : 1e-9,
        "lev"          : ilev,
        "thicken_contours" : [0.0,],
    },

    "T" : {
        "display"      : "Air temperature",
        "unit"         : r"\mathrm{K}",
        "var"          : "T_ZONAL_MM",
        "var_std"      : "T_ZONAL_MASTD",
        "cmap_mean"    : "bwr",
        "clevels_mean" : np.linspace(200, 350, 16),
        "clevels_diff" : np.linspace(-2, 2,  9),
        "clevels_resp" : np.linspace(-1, 1,  9),
        "clevels_resp_tick" : np.linspace(-1, 1, 5),
        "factor"       : 1.0,
        "lev"          : lev,
    },

    "U" : {
        "display"   : "Zonal Wind Velocity",
        "unit"      : r"\mathrm{m} \, / \, \mathrm{s}",
        "var"       : "U_ZONAL_MM",
        "var_std"   : "U_ZONAL_MASTD",
        "cmap_mean" : "bwr",
        "clevels_mean" : np.linspace(-200, 200, 41),
        "clevels_diff" : np.linspace(-2, 2,  9),
        "clevels_resp" : np.linspace(-1, 1,  9),
        "clevels_resp_tick" : np.linspace(-1, 1, 5),
        "factor"       : 1.0,
        "lev"          : lev,
    },


}

try: 
    os.makedirs(output_dir)

except:
    pass


plot_vars = ["T", "U"]

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

 


    if plot_type == "diff_OGCM":  # Then OGCM is plotted as contour over the rest plots
        heights = [1,] * (len(sim_casenames.keys())-1)
    elif plot_type == "diff_EXP":
        heights = [1,] * len(sim_casenames.keys())

    heights += [0.1,]
    widths = [1, ] * len(plot_vars)
    
    fig = plt.figure(figsize=(5 * len(plot_vars), 3 * (len(heights)-1)))
    spec = fig.add_gridspec(nrows=len(heights), ncols=len(widths), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3, left=0.2) 

    for (i, varname) in enumerate(plot_vars):
        
        plot_info = plot_infos[varname]


        ax = []
       
        factor = plot_info["factor"]

        cmap_diff = cm.get_cmap("bwr")
        cmap_diff.set_over("gold")
        cmap_diff.set_under("dodgerblue")

        if plot_type == "diff_OGCM": 
            clevels_diff  = plot_info["clevels_diff"]
        if plot_type == "diff_EXP": 
            clevels_diff  = plot_info["clevels_resp"]

        if "clevels_resp_tick" in plot_info:
            clevels_diff_tick = plot_info["clevels_resp_tick"]
        else:
            clevels_diff_tick = clevels_diff


        norm = mplt.colors.BoundaryNorm(boundaries=clevels_diff, ncolors=256)

        idx=0
        for exp_name, caseinfo in sim_casenames.items():

            label = exp_name
            lc = caseinfo["lc"]
            ls = caseinfo["ls"]
            row_idx = idx#caseinfo["col_idx"]
            idx+=1

            if plot_type == "diff_OGCM" and exp_name == "OGCM":
                continue

            _ax = fig.add_subplot(spec[row_idx, i])
            ax.append(_ax)
            
            #print(data["EXP"][exp_name][plot_info["var"]][:].shape)
            if plot_type == "diff_OGCM":
                raw_DAT = data["CTL"][exp_name]
                raw_REF = data["CTL"][ref_casename]
            
            elif plot_type == "diff_EXP":
                raw_DAT = data["EXP"][exp_name]
                raw_REF = data["CTL"][exp_name]

            _DAT = np.mean(raw_DAT[plot_info["var"]][rng, :, :], axis=(0,)) * factor
            _REF = np.mean(raw_REF[plot_info["var"]][rng, :, :], axis=(0,)) * factor
            _STD_DAT = np.mean(raw_DAT[plot_info["var_std"]][rng, :, :], axis=(0,)) * factor
            _STD_REF = np.mean(raw_REF[plot_info["var_std"]][rng, :, :], axis=(0,)) * factor

            _diff = _DAT - _REF

            _diff_abs = np.abs(_diff).flatten()
            max_chg_idx = np.argsort(_diff_abs)
            top_chg_idx = np.unravel_index(max_chg_idx[-1], _diff.shape)
            print("[%s] Max change: %.2e. (i, j) = (%d, %d). Max 5 points mean: %.2e" % (exp_name, _diff_abs[max_chg_idx[-1]], top_chg_idx[0], top_chg_idx[1], np.mean(_diff_abs[max_chg_idx[::-5]])))
            
            pooled_std = ((_STD_DAT**2.0 + _STD_REF**2.0) / 2.0 )**0.5
            student_t_score = _diff / pooled_std * ( nyears / 2.0 )**0.5
           
            threshold = 2.0 
            cut_t_score = np.abs(student_t_score)


            mappable_diff = _ax.contourf(lat, plot_info["lev"], _diff, clevels_diff, cmap=cmap_diff, extend="both", norm=norm)

            cs = _ax.contour(lat, plot_info["lev"], _REF, plot_info["clevels_mean"], colors='k', linewidths=1)
            labels = plt.clabel(cs, fmt="%d", fontsize=10)
            for l in labels:
                l.set_rotation(0)

            if "thicken_contours" in plot_info:
                _ = _ax.contour(lat, plot_info["lev"], _DAT, plot_info["thicken_contours"], colors='k', linewidths=2)


#            if plot_type == "diff_EXP":
#                cs = _ax.contourf(lat, plot_info["lev"], cut_t_score, [0.0, threshold, threshold+1], alpha=0, hatches=[None, None, '..', '..'], extend="both")

            if i == 0: 
                if plot_type == "diff_OGCM":
                    _ax.set_title("%s_CTL" % (label,))
                elif plot_type == "diff_EXP":
                    _ax.text(-0.32, 0.5, "RESP_%s" % (label,), transform=_ax.transAxes, ha="center", va="center", size=15, rotation=90)
             
               
            _ax.set_xlim([-90, 90])
            _ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            _ax.set_xticklabels([])
            _ax.set_yticks([1000, 750, 500, 250, 10])
            _ax.set_yticklabels([""]*5)
            _ax.invert_yaxis()

            
            if i == 0:
                _ax.set_yticklabels(["1000", "750", "500", "250", "10"], fontsize=10)
                _ax.set_ylabel("[ hPa ]", fontsize=10)
 

            if row_idx == len(heights) - 2:
                _ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"], fontsize=10)
                #_ax.set_xticklabels(["90S", "", "", "EQ", "", "", "90N"])
        
            if row_idx == 0:
                _ax.set_title("%s" % (plot_info["display"],), fontsize=15)
 
       
 
        cax = fig.add_subplot(spec[-1, i])
        cb_diff = fig.colorbar(mappable_diff, cax=cax, orientation="horizontal", ticks=clevels_diff_tick)
        cb_diff.ax.tick_params(labelsize=12)
        #cb_diff.set_label("%s [ $ %s $ ]" % (plot_info["display"], plot_info["unit"],))
        cb_diff.set_label("[ $ %s $ ]" % (plot_info["unit"],))



    plt.show()
    fig.savefig("%s/CTL_zmean_cx_%s_%s_col.png" % (output_dir, plot_type, ext), dpi=600)
    plt.close(fig)

