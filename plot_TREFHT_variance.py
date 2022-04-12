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


plot_type = ["plain", "diff", "resp"][1]
ref_casename = "OGCM"

domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "graph"
ENSO_file = "data/CTL_21-120/POP2_CTL/atm_analysis_ENSO.nc"


sim_casenames = getSimcases(["SOM", "MLM", "EMOM", "OGCM"])
sim_var = getSimVars(["TREFHT",])

with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

mask_lnd = mask == 1.0
data = {}

for scenario in ["CTL", "EXP"]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():

        casename = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/%s/%s" % (casename, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s => %s" % (casename, varname))
                var_mean = "%s_SM" % varname
                var_std  = "%s_SASTD" % varname
 
                var_mean = "%s_AM" % varname
                var_std  = "%s_AASTD" % varname
                
                data[scenario][exp_name][var_mean] = f.variables[var_mean][:, 0, :, :]
                data[scenario][exp_name][var_std]  = f.variables[var_std][:, 0, :, :]
                
            

plot_info = {
    "TREFHT" : {
        "display"   : "std(TREFHT)",
        "unit"      : "[${}^\\circ\\mathrm{C}$]",
        "cmap_mean" : "hot_r",
        "clevels_plain"       : np.linspace(0, 3, 11),
        "clevels_plain_tick"  : np.array([0, 1, 2, 3]),
        "clevels_diff" : np.linspace(-1, 1,  11) * 0.5,
        "clevels_resp" : np.linspace(-0.2, 0.2,  11),
        "factor"        : 1.0,
    },
}["TREFHT"]


try: 
    os.makedirs(output_dir)

except:
    pass


proj1 = ccrs.PlateCarree(central_longitude=180.0)
data_proj = ccrs.PlateCarree(central_longitude=0.0)
    
#proj1 = ccrs.NorthPolarStereo()
#data_proj = ccrs.NorthPolarStereo()

proj_kw = {
    'projection':proj1,
    'aspect' : 'auto',
}

plot_seasons = [0, 1, 2, 3]
plot_seasons = [1, 3,]
plot_seasons = [0]

fig = plt.figure(figsize=(6 * len(sim_casenames), 3 * len(plot_seasons)))
widths  = [0.05] + [1] * len(sim_casenames) + [0.05]
heights = [1,] * len(plot_seasons)
spec = fig.add_gridspec(nrows=len(plot_seasons), ncols=len(widths), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.2) 

    
ax = []
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


for i, season in enumerate(plot_seasons):

    j = 0

    for exp_name, caseinfo in sim_casenames.items():

        label = exp_name
        ax_idx = caseinfo["ax_idx"]

        print(label)

        _ax = fig.add_subplot(spec[i, j+1], **proj_kw)
        ax.append(_ax)

        j+=1
        var_mean = "%s_SM" % varname
        var_std  = "%s_SASTD" % varname

        var_mean = "%s_AM" % varname
        var_std  = "%s_AASTD" % varname


        _EXP = data["EXP"][exp_name][var_std][season, :, :] * factor
        _CTL = data["CTL"][exp_name][var_std][season, :, :] * factor
        _ref_CTL = data["CTL"][ref_casename][var_std][season, :, :] * factor

#        _EXP[mask_lnd] = np.nan
#        _CTL[mask_lnd] = np.nan
#        _ref_CTL[mask_lnd] = np.nan

        if plot_type == "plain":
            mappable = _ax.contourf(lon, lat, _CTL,  clevels_plain,  cmap=cmap_plain, transform=data_proj, extend="max")
            #mappable = _ax.contour(lon, lat, _CTL,  clevels_plain,  transform=data_proj)

        elif plot_type == "diff":

            if exp_name == "OGCM":
                mappable = _ax.contourf(lon, lat, _CTL,  clevels_plain,  cmap=cmap_plain, transform=data_proj, extend="both")
            else:
                mappable_diff = _ax.contourf(lon, lat, _CTL - _ref_CTL,  clevels_diff,  cmap=cmap_diff, transform=data_proj, extend="both")
 
        elif plot_type == "resp":
            mappable = _ax.contourf(lon, lat, _EXP - _CTL,  clevels_resp,  cmap=cmap_diff, transform=data_proj, extend="both")

           

        _ax.coastlines()

        if i==0:

            if plot_type == "plain":
                _ax.set_title("CTL_%s " % (label, ))

            elif plot_type == "diff":
                if exp_name == "OGCM":
                    _ax.set_title("CTL_%s " % (label, ))
                else:
                    _ax.set_title("CTL_%s - CTL_OGCM" % (label, ))

            elif plot_type == "resp":
                _ax.set_title("SIL_%s - CTL_%s" % (label, label))



for _ax in ax:
#    _ax.set_aspect('auto')
#    _ax.set_ylim([-90, 90])
#    _ax.set_yticks([])#[-90, -60, -30, 0, 30, 60, 90])
#    _ax.set_yticklabels([])
    #["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])

#            _ax.set_xlim([0, 360])
#            _ax.set_xticks(np.linspace(0,360,7), crs=proj1)
#            _ax.set_xticklabels([])#["0", "60E", "120E", "180", "120W", "60W", "0"])
    pass


#        fig.subplots_adjust(right=0.85)

cax = fig.add_subplot(spec[0, -1])
cb = fig.colorbar(mappable,  cax=cax, ticks=clevels_tick, orientation="vertical")
cb.set_label("%s %s" % (plot_info["display"], plot_info["unit"]))

if plot_type == "diff":
    cax_diff = fig.add_subplot(spec[i, 0])
    cb_diff = fig.colorbar(mappable_diff,  cax=cax_diff, ticks=clevels_diff_tick, orientation="vertical")
    cb_diff.set_label("%s bias %s" % (plot_info["display"], plot_info["unit"]))
    cax_diff.yaxis.set_ticks_position("left")
    cax_diff.yaxis.set_label_position("left")

fig.savefig("%s/TREFHT_variance_%s.png" % (output_dir, plot_type), dpi=300)
plt.show()
plt.close(fig)

