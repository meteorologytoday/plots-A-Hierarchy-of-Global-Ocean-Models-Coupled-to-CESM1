import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib as mplt
from matplotlib import cm
import matplotlib.ticker as mticker
import matplotlib.patches as patches
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


plot_type = ["plain", "diff", "resp"][2]
ref_casename = "OGCM"

domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "figures"

sim_casenames = getSimcases(["SOM", "MLM", "EMOM", "OGCM"])
sim_var = getSimVars(["CORR",])

with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

mask_lnd = mask == 1.0
data = {}

for scenario in ["CTL",]:

    data[scenario] = {}

    for exp_name, caseinfo in sim_casenames.items():

        casename = caseinfo[scenario]
        data[scenario][exp_name] = {}
        
        for varname, filename  in sim_var.items():

            filename = "data/hierarchy_statistics/%s/%s" % (casename, filename, )
            
            with Dataset(filename, "r") as f:
                print("%s" % (casename,))
                _d = f.variables[varname][:]

            
            #data[scenario][exp_name][varname]  = {
            #    "Mar-Apr-May" : np.mean(_d[2:5,  :, :], axis=0),
            #    "Sep-Oct-Nov" : np.mean(_d[8:11, :, :], axis=0),
            #}

            data[scenario][exp_name][varname]  = {
                "Dec-Jan-Feb" : np.mean(_d[(11,0,1),  :, :], axis=0),
                "Jun-Jul-Aug" : np.mean(_d[(5,6,7), :, :], axis=0),
            }


try: 
    os.makedirs(output_dir)

except:
    pass

boxes = [
    (0, "A", [20, 65], [-60, 60], -10),
    (0, "C", [-20, 10], [-20, 100], 15),
    (1, "C", [-20, 10], [-20, 100], 15),
    (1, "B", [-65, -35], [-175, 175], 15),
]


proj1 = ccrs.PlateCarree(central_longitude=180.0)
data_proj = ccrs.PlateCarree(central_longitude=0.0)

proj_kw = {
    'projection':proj1,
    'aspect' : 'auto',
}

#plot_seasons = ["Mar-Apr-May", "Sep-Oct-Nov"]
plot_seasons = ["Dec-Jan-Feb", "Jun-Jul-Aug"]

fig = plt.figure(figsize=(5 * len(plot_seasons), 3*len(sim_casenames)))
heights = [1] * len(sim_casenames)
widths  = [1,] * len(plot_seasons) + [0.05]
spec = fig.add_gridspec(nrows=len(heights), ncols=len(widths), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3, right=0.8)

    
ax = []

for i, season in enumerate(plot_seasons):

    for j, (exp_name, caseinfo) in enumerate(sim_casenames.items()):

        label = exp_name
        row_idx = caseinfo["col_idx"]

        print(label)

        _ax = fig.add_subplot(spec[row_idx, i], **proj_kw)
        ax.append(_ax)


        var_mean = "%s_AM" % varname
        var_std  = "%s_AASTD" % varname

        _CTL = data["CTL"][exp_name]["CORR"][season]
        _CTL[mask_lnd] = np.nan

        mappable = _ax.contourf(lon, lat, _CTL,  np.linspace(-0.8, 0.8, 9),  cmap=cm.get_cmap("bwr"), transform=data_proj, extend="both")
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

       
        for l, (_t_idx, box_label, lat_rng, lon_rng, text_pos) in enumerate(boxes):

            if _t_idx == i:

                _ax.add_patch(patches.Rectangle((lon_rng[0], lat_rng[0]), lon_rng[1] - lon_rng[0], lat_rng[1] - lat_rng[0], linewidth=1, edgecolor='black', facecolor='none', zorder=99))
                ha = "left" if text_pos >= 0 else "right"
                text_pos = lon_rng[0 if text_pos >= 0 else 1] + text_pos
                _ax.text(text_pos, np.mean(lat_rng), box_label, va="center", ha=ha, color="black")

        if i==0:
            _ax.set_ylabel("CTL_%s " % (label, ), fontsize=15, labelpad=30)
        
        if row_idx == 0:
            _ax.set_title(season, size=15)
        


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

cax = fig.add_subplot(spec[0, -1])
cb = fig.colorbar(mappable,  cax=cax, ticks=np.linspace(-.8, .8, 5), orientation="vertical")
cb.set_label("Correlation")

fig.savefig("%s/fig04_SSTA_correlation_col.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)

