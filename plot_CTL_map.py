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


plot_type = ["plain", "diff"][1]
ref_casename = "OGCM"

domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "graph"

sim_casenames = getSimcases(["SOM", "MLM", "EMOM", "OGCM"])
sim_var = getSimVars(["SST", "TREFHT", "PREC_TOTAL", "STRAT", "PSL", "vice"])

Re = 6371e3
with Dataset(domain_file, "r") as f:
    llat  = f.variables["yc"][:]
    llon  = f.variables["xc"][:]
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]
    dx = 2 * np.pi * np.cos(f.variables["yc"][:] * np.pi / 180.0) * Re / len(lon)
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

lnd_mask_idx = (mask == 1.0)

data = {}

for scenario in ["CTL",]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():

        casename = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/%s/%s" % (casename, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s => %s" % (casename, varname))
                var_mean = "%s_MM" % varname
                var_std  = "%s_MASTD" % varname
                
                data[scenario][exp_name][var_mean] = f.variables[var_mean][:, 0, :, :]
                data[scenario][exp_name][var_std]  = f.variables[var_std][:, 0, :, :]
                
                if varname == "TREFHT":
                    data[scenario][exp_name][var_mean] -= 273.15


# This section I just wanna compute the mean difference of tropical
# precipitation numerically.

valid_idx = (np.abs(llat) > 30.0)
    
for varname in ["PREC_TOTAL", "SST"]:

    print("# varname: %s" % varname)
    for exp_name, caseinfo in sim_casenames.items():

        _REF = data["CTL"]["OGCM"]["%s_MM" % varname]
        _CTL = data["CTL"][exp_name]["%s_MM" % varname]
        _DIFF = (_CTL - _REF).mean(axis=0)
        _diff_weighted = np.sum(_DIFF[valid_idx] * area[valid_idx]) / np.sum(area[valid_idx])

        print("Mean difference: %s => %.2e" % (exp_name, _diff_weighted,) )


plot_infos = {

    "LHFLX" : {
        "display"   : "LHFLX",
        "unit"      : r"[$\mathrm{W} / \mathrm{m}^2$]",
        "cmap_mean" : "GnBu",
        "cmap_diff" : "BrBG",
        "clevels_plain"  : np.linspace(0, 10, 41),
        "clevels_plain_tick"  : np.array([0, 2, 4, 6, 8, 10]),
        "clevels_diff" : np.linspace(-1, 1,  11),
        "factor"       : 1.0,
    },

    "SHFLX" : {
        "display"   : "SHFLX",
        "unit"      : r"[$\mathrm{W} / \mathrm{m}^2$]",
        "cmap_mean" : "GnBu",
        "cmap_diff" : "BrBG",
        "clevels_plain"  : np.linspace(0, 10, 41),
        "clevels_plain_tick"  : np.array([0, 2, 4, 6, 8, 10]),
        "clevels_diff" : np.linspace(-1, 1,  11),
        "factor"       : 1.0,
    },




    "PREC_TOTAL" : {
        "display"   : "Precipitation rate",
        "unit"      : "[mm / day]",
        "cmap_mean" : "GnBu",
        "cmap_diff" : "BrBG",
        "clevels_plain"  : np.linspace(0, 10, 41),
        "clevels_plain_tick"  : np.array([0, 2, 4, 6, 8, 10]),
        "clevels_diff" : np.linspace(-4, 4,  11),
        "factor"       : 86400.0 * 1000.0,
    },


    "STRAT" : {
        "display"   : r"$T_\mathrm{st}$",
        "unit"      : "[degC]",
        "cmap_mean" : "rainbow",
        "clevels_plain"       : np.linspace(0, 20, 41),
        "clevels_plain_tick"  : np.array([0, 5, 10, 15, 20]),
        "clevels_diff" : np.linspace(-2, 2,  11),
        "factor"        : 1.0,
    },

    "SST" : {
        "display"   : "SST",
        "unit"      : "[degC]",
        "cmap_mean" : "rainbow",
        "clevels_plain"       : np.linspace(-2, 30, 33),
        "clevels_plain_tick"  : np.array([-2, 0.0, 5, 10, 15, 20, 25, 30]),
        "clevels_diff" : np.linspace(-2, 2,  11),
        "factor"        : 1.0,
    },


    "TREFHT" : {
        "display"   : "SAT",
        "unit"      : "[degC]",
        "cmap_mean" : "rainbow",
        "clevels_plain"       : np.linspace(-10, 30, 41),
        "clevels_plain_tick"  : np.array([-10, 0, 10, 20, 30]),
        "clevels_diff" : np.linspace(-2, 2,  11),
        "factor"        : 1.0,
    },

    "PSL" : {
        "display"   : r"Sea-level pressure",
        "unit"      : "[hPa]",
        "cmap_mean" : "gnuplot",
        "clevels_plain"  : np.linspace(980, 1020, 41),
        "clevels_plain_tick"  : np.array([980, 990, 1000, 1010, 1020]),
        "clevels_diff" : np.linspace(-5,   5, 11),
        "factor"       : 1e-2,
    },

    "h_ML" : {
        "display"   : "MLT",
        "unit"      : "[m]",
        "cmap_mean" : "GnBu",
        "clevels_plain"  : np.linspace(0, 300, 31),
        "clevels_diff" : np.linspace(-20, 20, 11),
        "factor"        : 1.0,
    },

    "aice" : {
        "display"   : "SIA",
        "unit"      : "[%]",
        "cmap_mean" : "Blues",
        "cmap_diff" : "BrBG",
        "clevels_plain"       : np.linspace(1, 100, 100),
        "clevels_plain_tick"  : np.array([0, 20, 40, 60, 80, 100]),
        "clevels_diff" : np.linspace(-20, 20,  11),
        "factor"        : 100.0,
    },

    "vice" : {
        "display"   : "SIT",
        "unit"      : "[m]",
        "cmap_mean" : "Blues",
        "cmap_diff" : "BrBG",
        "clevels_plain"       : np.linspace(0, 2, 51),
        "clevels_plain_tick"  : np.array([0, 0.5, 1, 1.5, 2,]) ,
        "clevels_diff" : np.linspace(-1, 1,  11),
        "factor"        : 1,
    },


}


try: 
    os.makedirs(output_dir)

except:
    pass


proj1 = ccrs.PlateCarree(central_longitude=180.0)
data_proj = ccrs.PlateCarree(central_longitude=0.0)

proj_kw = {
    'projection':proj1,
    'aspect' : 'auto',
}


plot_vars = ["SST", "TREFHT", "PREC_TOTAL", "STRAT", "PSL", "vice"]
plot_vars = ["SST",]

for m in [4,]:#[0,2,4,]:#range(5):

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

        
    print("Time : ", ext)
 
    # Original
    fig = plt.figure(figsize=(5*len(sim_casenames), 3*len(plot_vars)))
    widths  = [0.05] + [1] * len(sim_casenames) + [0.05]
    heights = [1,] * len(plot_vars)
    spec = fig.add_gridspec(nrows=len(plot_vars), ncols=len(sim_casenames)+2, width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.2) 


    for (i, varname) in enumerate(plot_vars):
        
        plot_info = plot_infos[varname]

        ax = []
       
        factor = plot_info["factor"]

        if "cmap_diff" in plot_info:
            cmap_diff = cm.get_cmap(plot_info["cmap_diff"])
        else:
            cmap_diff = cm.get_cmap("bwr")
            
        clevels_diff  = plot_info["clevels_diff"]
        clevels_plain  = plot_info["clevels_plain"]
        cmap_plain  = plot_info["cmap_mean"]
        
        if "clevels_plain_tick" in plot_info:
            clevels_tick  = plot_info["clevels_plain_tick"]
        else:
            clevels_tick  = plot_info["clevels_plain"]


        if "clevels_diff_tick" in plot_info:
            clevels_diff_tick  = plot_info["clevels_diff_tick"]
        else:
            clevels_diff_tick  = plot_info["clevels_diff"]


        print("Plotting ", varname)

#        fig.suptitle("[%s] %s" % (ext, plot_info["display"]))

        for exp_name, caseinfo in sim_casenames.items():


            if exp_name == "SOM" and varname == "STRAT":
                continue

            label = exp_name
            lc = caseinfo["lc"]
            ls = caseinfo["ls"]
            ax_idx = caseinfo["ax_idx"]

            _ax = fig.add_subplot(spec[i, ax_idx[1]+1], **proj_kw)
            ax.append(_ax)
            
            var_mean = "%s_MM" % varname
            var_std  = "%s_MASTD" % varname
 

            #_EXP = np.mean(data["EXP"][exp_name][var_mean][rng, :, :], axis=(0,)) * factor
            _CTL = np.mean(data["CTL"][exp_name][var_mean][rng, :, :], axis=(0,)) * factor
            _ref_CTL = np.mean(data["CTL"][ref_casename][var_mean][rng, :, :], axis=(0,)) * factor

            if varname == "SST":
                _CTL -= 273.15
                _ref_CTL -= 273.15

                _CTL[_CTL < -200] = np.nan
                _ref_CTL[_ref_CTL < -200] = np.nan

            #_STD_EXP = np.mean(data["EXP"][exp_name][var_std][rng, :, :], axis=(0,)) * factor
            _STD_CTL = np.mean(data["CTL"][exp_name][var_std][rng, :, :], axis=(0,)) * factor


            #_diff = _EXP - _CTL
            #_diff_mean = 0#area_mean(_diff, area)
            #_diff -= _diff_mean

            #pooled_std = ((_STD_CTL**2.0 + _STD_EXP**2.0) / 2.0 )**0.5
            #student_t_score = _diff / pooled_std * ( nyears / 2.0 )**0.5
#            student_t_score[lnd_mask_idx] = 300

            #threshold = 2.0 
            #cut_t_score = np.abs(student_t_score)

            #mappable_diff = _ax.contourf(lon, lat, _diff,  clevels_diff,  cmap=cmap_diff, extend="both", transform=data_proj)

            if plot_type == "plain":
                mappable = _ax.contourf(lon, lat, _CTL,  clevels_plain,  cmap=cmap_plain, transform=data_proj, extend="both")

            elif plot_type == "diff":
                if exp_name == "OGCM":
                    mappable = _ax.contourf(lon, lat, _CTL,  clevels_plain,  cmap=cmap_plain, transform=data_proj, extend="both")
                else:
                    mappable_diff = _ax.contourf(lon, lat, _CTL - _ref_CTL,  clevels_diff,  cmap=cmap_diff, transform=data_proj, extend="both")
                

            _ax.coastlines()

#            cs = _ax.contourf(lon, lat, cut_t_score, [0.0, threshold, threshold+1], alpha=0, hatches=[None, None, '...', '...'], extend="both", transform=data_proj)
#            artists, labels = cs.legend_elements()
#            _ax.legend(artists, labels, handleheight=2)

            #_ax.set_title("%s diff ( $\\Delta_{\\mathrm{mean}} = %.2f $ )" % (label, _diff_mean, ))

            if i==0:

                if plot_type == "plain":
                    _ax.set_title("CTL_%s " % (label, ))

                elif plot_type == "diff":
                    if exp_name == "OGCM":
                        _ax.set_title("CTL_%s " % (label, ))
                    else:
                        _ax.set_title("CTL_%s - CTL_OGCM" % (label, ))

            #if ax_idx[1] == 0:
            #    _ax.text(-0.15, 0.5, plot_info["display"],  rotation=0, va="center", ha="center", transform=_ax.transAxes)

        for _ax in ax:
            _ax.set_aspect('auto')
            _ax.set_ylim([-90, 90])
            _ax.set_yticks([])#[-90, -60, -30, 0, 30, 60, 90])
            _ax.set_yticklabels([])
            #["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
        
#            _ax.set_xlim([0, 360])
#            _ax.set_xticks(np.linspace(0,360,7), crs=proj1)
#            _ax.set_xticklabels([])#["0", "60E", "120E", "180", "120W", "60W", "0"])


 
#        fig.subplots_adjust(right=0.85)

        cax = fig.add_subplot(spec[i, -1])
        cb = fig.colorbar(mappable,  cax=cax, ticks=clevels_tick, orientation="vertical")
        cb.set_label("%s %s" % (plot_info["display"], plot_info["unit"]))

        if plot_type == "diff":
            cax_diff = fig.add_subplot(spec[i, 0])
            cb_diff = fig.colorbar(mappable_diff,  cax=cax_diff, ticks=clevels_diff_tick, orientation="vertical")
            cb_diff.set_label("%s bias %s" % (plot_info["display"], plot_info["unit"]))
            cax_diff.yaxis.set_ticks_position("left")
            cax_diff.yaxis.set_label_position("left")

        #cb_plain.ax.set_ylabel("[m]", rotation=90, labelpad=2)
        #cb_diff.ax.set_ylabel("[m]", rotation=90, labelpad=2)
#    cb_std_diff.ax.tick_params(labelsize=30)



    fig.savefig("%s/CTL_map_%s_%s.png" % (output_dir, plot_type,ext), dpi=300)
    plt.show()
    plt.close(fig)

