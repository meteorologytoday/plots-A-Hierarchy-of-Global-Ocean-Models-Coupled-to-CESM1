import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib as mplt
#mplt.use('Agg')

from matplotlib import cm
import matplotlib.ticker as mticker
import matplotlib.patches as patches

from quick_tools import *


from matplotlib import rc

import os

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

def rmSignal(data, signal):

    if len(data) != len(signal):
        print("Length of data=%d length of signal=%d" % (len(data), len(signal)))
        raise Exception("Unequal Length.")
    

    data_a = data - np.mean(data)
    signal_a = signal - np.mean(signal)

    return data_a -( np.inner(data_a, signal_a) / np.inner(signal_a, signal_a)) * signal_a




ref_casename = "OGCM*"

domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "figures"


plot_t_idx = [1, 3]
plot_t_title = ["Jun-Jul-Aug", "Dec-Jan-Feb"]

plot_vars = ["TREFHT",]
plot_types = ["diff"]
#plot_type = ["plain", "diff", "resp"][0]



avg_time = ["annual", "season"][1]

sim_casenames = getSimcases(["SOM", "MLM", "EMOM", "OGCM"])
sim_var = getSimVars(plot_vars)

#plot_exp_names = ["SOM", "MLM", "EMOM", "OGCM*", "OGCM"]
plot_exp_names = ["EMOM", "MLM", "SOM"]

with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

mask_lnd = mask == 1.0
data = {}

if avg_time == "season":
    var_mean_temp = "%s_SM"
    var_std_temp  = "%s_SASTD"

elif avg_time == "annual":
    var_mean_temp = "%s_AM"
    var_std_temp  = "%s_AASTD"

for scenario in ["CTL", "EXP"]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():

        casename = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/hierarchy_statistics/%s/%s" % (casename, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s => %s" % (casename, varname))

                var_mean = var_mean_temp % varname
                var_std  = var_std_temp  % varname
                        
                data[scenario][exp_name][var_mean] = f.variables[var_mean][:, 0, :, :]
                data[scenario][exp_name][var_std]  = f.variables[var_std][:, 0, :, :]

data["CTL"]["OGCM*"] = {} 
sim_casenames["OGCM*"] = {}
for varname in plot_vars:
 
    with Dataset("data_extra/OGCM_%sA_Statistics_remove_ENSO_annual.nc" % (varname,), "r") as f:
        OGCM_VARA_noENSO_std = f.variables["VARA_noENSO_STD"][:]


    data["CTL"]["OGCM*"]["%s_SASTD" % varname] = OGCM_VARA_noENSO_std 


plot_infos = {

    "SST" : {
        "display"   : "SSTA",
        "unit"      : "[${}^\\circ\\mathrm{C}$]",
        "cmap_mean" : "OrRd",
        "clevels_plain"       : np.linspace(0, 1, 11),
        "clevels_plain_tick"  : np.array([0, 1]),
        "clevels_diff" : np.linspace(-1, 1,  11) * 0.5,
        "clevels_resp" : np.linspace(-0.2, 0.2,  11),
        "factor"        : 1.0,
    },


    "PREC_TOTAL" : {
        "display"   : "Precipitation rate",
        "unit"      : "[mm / day]",
        "cmap_mean" : "GnBu",
        "cmap_diff" : "BrBG",
        "clevels_plain"  : np.linspace(0, 10, 41),
        "clevels_plain_tick"  : np.array([0, 2, 4, 6, 8, 10]),
        "clevels_diff" : np.linspace(-1, 1,  11),
        "clevels_resp" : np.linspace(-1, 1,  11),
        "factor"       : 86400.0 * 1000.0,
    },



    "PSL" : {
        "display"   : r"SLPA",
        "unit"      : "[hPa]",
        "cmap_mean" : "BuGn",
        "clevels_plain"  : np.linspace(0, 10, 11),
        "clevels_plain_tick"  : np.array([0, 10]),
        "clevels_diff" : np.linspace(-1,   1, 11),
        "clevels_resp" : np.linspace(-5,   5, 11),
        "factor"       : 1e-2,
    },

    "TREFHT" : {
        "display"   : "SAT",
        "unit"      : "[degC]",
        "cmap_mean" : "OrRd",
        "clevels_plain"       : np.linspace(0, 5, 11),
        "clevels_plain_tick"  : np.array([0, 5]),
        "clevels_diff" : np.linspace(-1, 1,  11) * 0.5,
        "clevels_resp" : np.linspace(-2, 2,  11),
        "factor"        : 1.0,
    },


}


try: 
    os.makedirs(output_dir)

except:
    pass

boxes = [
#    (1, [-80, -50], [-175, 175], -5),
#    (3, [30, 60], [50, 110], -5),
]


proj1 = ccrs.PlateCarree(central_longitude=180.0)
data_proj = ccrs.PlateCarree(central_longitude=0.0)

proj_kw = {
    'projection':proj1,
    'aspect' : 'auto',
}

fig = plt.figure(figsize=(5 * len(plot_t_idx) * len(plot_vars), 3 * len(plot_exp_names)))
heights = [1] * len(plot_exp_names)
widths  = [1,] * (len(plot_t_idx) * len(plot_vars)) + [0.05,]
spec = fig.add_gridspec(nrows=len(heights), ncols=len(widths), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.2, right=0.8) 

    
ax = []


row_skip_per_var = len(plot_t_idx)

prev_plot_type = ""
for k, varname in enumerate(plot_vars):

    plot_info = plot_infos[varname] 
    plot_type = plot_types[k]
    
    show_title = prev_plot_type != plot_type
    prev_plot_type = plot_type

    factor = plot_info["factor"]
    if "cmap_diff" in plot_info:
        cmap_diff = cm.get_cmap(plot_info["cmap_diff"])
    else:
        cmap_diff = cm.get_cmap("bwr")
        
    cmap_diff = cm.get_cmap("bwr")
            
    clevels_diff  = plot_info["clevels_diff"]
    clevels_resp  = plot_info["clevels_resp"]
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

    if "clevels_resp_tick" in plot_info:
        clevels_resp_tick  = plot_info["clevels_resp_tick"]
    else:
        clevels_resp_tick  = plot_info["clevels_resp"]


    for i, t_idx in enumerate(plot_t_idx):

        j = 0

        for exp_name in plot_exp_names:

            caseinfo = sim_casenames[exp_name]

            label = exp_name

            print(label)
            print(k * row_skip_per_var, "; ", j)

            _ax = fig.add_subplot(spec[j, k * row_skip_per_var + i], **proj_kw)
            ax.append(_ax)



            var_mean = var_mean_temp % varname
            var_std  = var_std_temp  % varname
               
            _CTL = data["CTL"][exp_name][var_std][t_idx, :, :] * factor
            _ref_CTL = data["CTL"][ref_casename][var_std][t_idx, :, :] * factor

            # temporary compute the change of SST variability in percentage
            #_ref_SOM_CTL = data["CTL"]["SOM"][var_std][t_idx, :, :] * factor

            if not (varname in ["PSL", "TREFHT", "PREC_TOTAL"]):
                _CTL[mask_lnd] = np.nan
                _ref_CTL[mask_lnd] = np.nan

            if plot_type == "plain":
                mappable = _ax.contourf(lon, lat, _CTL,  clevels_plain,  cmap=cmap_plain, transform=data_proj, extend="max")

            elif plot_type == "diff":

                if exp_name in ["OGCM", "OGCM*"]:
                    mappable = _ax.contourf(lon, lat, _CTL,  clevels_plain,  cmap=cmap_plain, transform=data_proj, extend="max")
                else:
                    mappable_diff = _ax.contourf(lon, lat, _CTL - _ref_CTL,  clevels_diff,  cmap=cmap_diff, transform=data_proj, extend="both")
     
            elif plot_type == "resp":
                mappable = _ax.contourf(lon, lat, _EXP - _CTL,  clevels_resp,  cmap=cmap_diff, transform=data_proj, extend="both")

               

            _ax.coastlines()
            gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.3, linestyle='-')
            
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
            gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

            gl.xlabel_style = {'size': 8, 'color': 'black', 'ha':'center'}
            gl.ylabel_style = {'size': 8, 'color': 'black', 'ha':'right'}

            
            if j == 0:
                _ax.set_title(plot_t_title[i])

            if i==0 and show_title:

                if plot_type == "plain":
                    _ax.set_ylabel("CTL_%s " % (label, ))

                elif plot_type == "diff":
                    if exp_name in ["OGCM", "OGCM*"]:
                        _ax.set_ylabel("CTL_%s " % (label, ))
                    else:
                        _ax.set_ylabel("CTL_%s bias" % (label, ), fontsize=15, labelpad=30)

                elif plot_type == "resp":
                    _ax.set_ylabel("SIL_%s - CTL_%s" % (label, label), labelpad=8)
            
            j+=1

            for l, (_t_idx, lat_rng, lon_rng, text_pos) in enumerate(boxes):
                if _t_idx == t_idx:

                    _ax.add_patch(patches.Rectangle((lon_rng[0], lat_rng[0]), lon_rng[1] - lon_rng[0], lat_rng[1] - lat_rng[0], linewidth=1, edgecolor='black', facecolor='none', zorder=99))
                    ha = "left" if text_pos >= 0 else "right"
                    text_pos = lon_rng[0 if text_pos >= 0 else 1] + text_pos
                    _ax.text(text_pos, np.mean(lat_rng), "%s" % ("ABCDEFG"[l]), va="center", ha=ha, color="black")
     
    #cax = fig.add_subplot(spec[k*row_skip_per_var:(k+1)*row_skip_per_var, -1])
    #cb = fig.colorbar(mappable,  cax=cax, ticks=clevels_tick, orientation="vertical")
    #cb.set_label("std(%s) %s" % (plot_info["display"], plot_info["unit"]), fontsize=15)

    if plot_type == "diff":
        cax_diff = fig.add_subplot(spec[0, -1])
        #cb_diff = fig.colorbar(mappable_diff,  cax=cax_diff, ticks=clevels_diff_tick, orientation="vertical")
        cb_diff = fig.colorbar(mappable_diff,  cax=cax_diff, orientation="vertical")
        cb_diff.set_label("%s variability\nbias %s" % (plot_info["display"], plot_info["unit"]), fontsize=15)


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

fig.savefig("%s/fig03_ctl-variability_SAT_col.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)

