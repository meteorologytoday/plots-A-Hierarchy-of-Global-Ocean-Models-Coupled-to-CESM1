import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib as mplt
import matplotlib.ticker as mticker
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
output_dir = "graph_qflx"

loaded_ocn_models = ["SOM", "MLM", "EMOM"]

Re = 6371e3
with Dataset(domain_file, "r") as f:
    llat  = f.variables["yc"][:]
    llon  = f.variables["xc"][:]
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]
    dx = 2 * np.pi * Re * np.cos(lat * np.pi / 180.0) / len(lon)
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]

lnd_mask_idx = (mask == 1.0)
rho = 1026.0
c_p = 3996.0

dz = None
z_rng = slice(0, 20)
data = {}
for ocn_model in loaded_ocn_models:

    _tmp = {}
    qflx_filename = "data/qflx/f09/qflx_%s.nc" % (ocn_model, )
        
    with Dataset(qflx_filename, "r") as f:
        print("Loading file: %s" % (qflx_filename,) )

        if dz is None:
            dz = f.variables["dz_cT"][z_rng, 0, 0]
 
        for varname in ["QFLXT"]: 
            _tmp[varname] = np.sum(f.variables[varname][:, z_rng, :, :] * dz[None, :, None, None], axis=1) * rho * c_p

    data[ocn_model] = _tmp

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

#######################################
ocn_models = ["EMOM", "MLM", "SOM"]
seasons = [
    ("Mar-Apr-May", [ 2, 3, 4], ),
    ("Jun-Jul-Aug", [ 5, 6, 7], ),
    ("Sep-Oct-Nov", [ 8, 9,10], ),
    ("Dec-Jan-Feb", [11, 0, 1], ),
]

#seasons = [
#    ("Annual", slice(None), ),
#]


heights = [1] * len(ocn_models)
widths  = [1] * len(seasons) + [0.05]
fig = plt.figure(figsize=(5*len(seasons), 3*len(ocn_models)))
spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3, right=0.8) 

ax = []
for i, ocn_model in enumerate(ocn_models):
        
    print("Plotting ", ocn_model)

    for j, (season, t_rng) in enumerate(seasons):

        print("Season: ", season)

        qflx = np.mean(data[ocn_model]["QFLXT"][t_rng, :, :], axis=0)

        _ax = fig.add_subplot(spec[i, j], **proj_kw)
        ax.append(_ax)
       

           
        cmap = cm.get_cmap("bwr")
        clev = np.linspace(-200, 200, 11)
        clevticks = np.linspace(-200, 200, 11)
        mappable = _ax.contourf(lon, lat, qflx,  clev,  cmap=cmap, transform=data_proj, extend="both")

        if i==0:
            _ax.set_title(season)

        if j==0:
            _ax.text(-0.1, 0.5, ocn_model, transform=_ax.transAxes, va="center", ha="right", rotation=90) 
            
    if i == 0:
        cax = fig.add_subplot(spec[0, -1])
        cb = fig.colorbar(mappable,  cax=cax, ticks=clevticks, orientation="vertical")
        cb.set_label("Integrated flux correction [ $ \\mathrm{W} / \\mathrm{m}^2$ ] ")
        


for _ax in ax:
    
    _ax.coastlines()
    _ax.add_feature(cfeature.LAND, color="#cccccc")
    _ax.set_aspect('auto')


    gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.3, linestyle='-')
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([0, 60, 120, 180, -120, -60])
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {'size': 8, 'color': 'black', 'ha':'center'}
    gl.ylabel_style = {'size': 8, 'color': 'black', 'ha':'right'}
    
    #_ax.set_extent([100, 270, -30, 30], crs=ccrs.PlateCarree())
    _ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())

fig.savefig("%s/qflx_analysis.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)

