import cartopy.crs as ccrs
import matplotlib as mplt
from matplotlib import cm

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


nyears = 50
dof = 2 *nyears - 2 # degree of freedom

domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"

atm_lev_file = "CESM_domains/atm_lev.nc"
atm_ilev_file = "CESM_domains/atm_ilev.nc"

output_dir = "graph"

sim_casenames = {

    "SOM" : {
        "CTL": "CTL_16-20/paper2021_SOM_CTL",
        "EXP": "EXP_61-80/paper2021_SOM_EXP",
        "lc" : "orangered",
        "ls" : "--",
        "ax_idx" : (0, 0),
    },


    "MLM" : {
        "CTL": "CTL_16-20/paper2021_MLM_CTL",
        "EXP": "EXP_61-80/paper2021_MLM_EXP",
        "lc" : "orangered",
        "ls" : "--",
        "ax_idx" : (0, 1),
    },

    
    "EMOM" : {
        "CTL": "CTL_16-20/paper2021_EMOM_CTL_coupled",
        "EXP": "EXP_61-80/paper2021_EMOM_EXP",
        "lc" : "dodgerblue",
        "ls" : "--",
        "ax_idx" : (0, 2),
    },

    "OGCM" : {
        "CTL": "CTL_16-20/paper2021_CTL_POP2",
        "EXP": "EXP_61-80/paper2021_EXP_POP2",
        "lc" : "black",
        "ls" : "-",
        "ax_idx" : (0, 3),
    },

}

sim_var = {
#    "psi" : "atm_analysis12_psi.nc",
    "T"   : "atm_analysis_mean_anomaly_T_zm.nc",
    "U"   : "atm_analysis_mean_anomaly_U_zm.nc",
}

Re = 6371e3
with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]

with Dataset(atm_lev_file, "r") as f:
    lev  = f.variables["lev"][:]

with Dataset(atm_ilev_file, "r") as f:
    ilev  = f.variables["ilev"][:]



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
        "display"      : "temperature",
        "unit"         : r"\mathrm{K}",
        "var"          : "T_ZONAL_MM",
        "var_std"      : "T_ZONAL_MASTD",
        "cmap_mean"    : "bwr",
        "clevels_mean" : np.linspace(200, 350, 16),
        "clevels_diff" : np.concatenate(([-2], np.linspace(-0.5, 0.5,  11), [2])),
        "factor"       : 1.0,
        "lev"          : lev,
    },

    "U" : {
        "display"   : "zonal wind",
        "unit"      : r"\mathrm{m} \, / \, \mathrm{s}",
        "var"       : "U_ZONAL_MM",
        "var_std"   : "U_ZONAL_MASTD",
        "cmap_mean" : "bwr",
        "clevels_mean" : np.linspace(-200, 200, 41),
        "clevels_diff" : np.linspace(-1, 1,  11),
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

 
    fig = plt.figure(constrained_layout=True, figsize=(20, 15))
    widths  = [1, 1, 1, 1, .05]
    heights = [1, ] * len(plot_vars)
    spec = fig.add_gridspec(nrows=len(plot_vars), ncols=5, width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.2) 

    for (i, varname) in enumerate(plot_vars):
        
        plot_info = plot_infos[varname]


        ax = []
       
        factor = plot_info["factor"]

        cmap_diff = cm.get_cmap("bwr")
        cmap_diff.set_over("gold")
        cmap_diff.set_under("dodgerblue")
        clevels_diff  = plot_info["clevels_diff"]
        norm = mplt.colors.BoundaryNorm(boundaries=clevels_diff, ncolors=256)



#        fig.suptitle("[%s] %s" % (ext, plot_info["display"]))
#        fig.suptitle("Change of %s" % (plot_info["display"],))


        for exp_name, caseinfo in sim_casenames.items():

            label = exp_name
            lc = caseinfo["lc"]
            ls = caseinfo["ls"]
            ax_idx = caseinfo["ax_idx"]

            _ax = fig.add_subplot(spec[i, ax_idx[1]])
            ax.append(_ax)
            
            #print(data["EXP"][exp_name][plot_info["var"]][:].shape)
            _EXP = np.mean(data["EXP"][exp_name][plot_info["var"]][rng, :, :], axis=(0,)) * factor
            _CTL = np.mean(data["CTL"][exp_name][plot_info["var"]][rng, :, :], axis=(0,)) * factor
            
            _diff = _EXP - _CTL

            _diff_abs = np.abs(_diff).flatten()
            max_chg_idx = np.argsort(_diff_abs)
            top_chg_idx = np.unravel_index(max_chg_idx[-1], _diff.shape)
            print("[%s] Max change: %.2e. (i, j) = (%d, %d). Max 5 points mean: %.2e" % (exp_name, _diff_abs[max_chg_idx[-1]], top_chg_idx[0], top_chg_idx[1], np.mean(_diff_abs[max_chg_idx[::-5]])))
            
             
#            for k in range(10):
#                max_chg_idx = np.unravel_index(np.argmax(np.abs(_diff)), _diff.shape)
#            print("[%s] Max change: %.2e. (i, j) = (%d, %d)" % (exp_name, max_chg, max_chg_idx[0], max_chg_idx[1]))
            
            _STD_CTL = np.mean(data["CTL"][exp_name][plot_info["var_std"]][rng, :, :], axis=(0,)) * factor
            _STD_EXP = np.mean(data["EXP"][exp_name][plot_info["var_std"]][rng, :, :], axis=(0,)) * factor

            pooled_std = ((_STD_CTL**2.0 + _STD_EXP**2.0) / 2.0 )**0.5
            student_t_score = _diff / pooled_std * ( nyears / 2.0 )**0.5
           
            threshold = 2.0 
            cut_t_score = np.abs(student_t_score)


            mappable_diff = _ax.contourf(lat, plot_info["lev"], _diff, clevels_diff, cmap=cmap_diff, extend="both", norm=norm)

            cs = _ax.contour(lat, plot_info["lev"], _CTL, plot_info["clevels_mean"], colors='k', linewidths=1)

            if "thicken_contours" in plot_info:
                _ = _ax.contour(lat, plot_info["lev"], _CTL, plot_info["thicken_contours"], colors='k', linewidths=2)

            cs = _ax.contourf(lat, plot_info["lev"], cut_t_score, [0.0, threshold, threshold+1], alpha=0, hatches=[None, None, '..', '..'], extend="both")

#            artists, labels = cs.legend_elements()
#            _ax.legend(artists, labels, handleheight=2)

            #CS = _ax.contour(lat, plot_info["lev"], student_t_score, [0.0, 0.5, 1.0, 1.5])
            #_ax.clabel(CS, inline=1, fontsize=10)
            if i == 0: 
                _ax.set_title(label)
             
               
            _ax.set_xlim([-90, 90])
            _ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            _ax.set_xticklabels([])
            _ax.set_yticks([1000, 750, 500, 250, 0])
            _ax.set_yticklabels([""]*5)
            _ax.invert_yaxis()

            if ax_idx[1]==0:
                _ax.set_yticklabels(["1000", "750", "500", "250", "0"])
                _ax.set_ylabel("[ hPa ]")
 

            #_ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
            _ax.set_xticklabels(["90S", "", "", "EQ", "", "", "90N"])
 
       
 
        #fig.subplots_adjust(left=0.1, right=0.9)

        cax = fig.add_subplot(spec[i, -1])
        cb_diff = fig.colorbar(mappable_diff, cax=cax, orientation="vertical", ticks=clevels_diff)
        cb_diff.set_label("[ $ %s $ ]" % (plot_info["unit"],))
        #cb_obs.ax.set_ylabel("[m]", rotation=90, labelpad=2)
        #cb_diff.ax.set_ylabel("[m]", rotation=90, labelpad=2)
#    cb_std_diff.ax.tick_params(labelsize=30)



plt.show()
fig.savefig("graph/diff_zmean_cx.png", dpi=600)
plt.close(fig)

