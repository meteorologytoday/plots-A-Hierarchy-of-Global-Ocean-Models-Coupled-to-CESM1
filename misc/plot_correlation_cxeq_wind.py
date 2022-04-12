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

domain_file = "CESM_domains/domain.lnd.fv0.9x1.25_gx1v6.090309.nc"
output_dir = "graph"
sim_casenames = ["SOM", "MLM", "EMOM", "OGCM"]

with Dataset(domain_file, "r") as f:
    lat  = f.variables["yc"][:, 0]
    lon  = f.variables["xc"][0, :]
    mask = f.variables["mask"][:]
    area = f.variables["area"][:]


data = {}

lon_rng = np.logical_and(lon >= 320, lon <= 345)
lat_rng = np.logical_and(lat >= -30, lat <=  30)

#print("lon_rng: ", lon_rng)
#print("lat_rng: ", lat_rng)


loaded_years = 100

t_rngs = [ ("JJA", slice(1, loaded_years*4, 4)), ("DJF", slice(3, loaded_years*4, 4)) ]

for sim_casename in sim_casenames:

    _data = {}
    
    for varname in ["TAUX", "TAUY", "PRECC", "SST"]:

        _data_var = {}
        filename = "output_1/%s_CTL/%s.nc" % (sim_casename, varname)
         
        print("Loading file: %s" % (filename,))
        with Dataset(filename, "r") as f:
            varname_used = "%s_SA" % (varname,)
            t_rng = slice(0)
            for (tlabel, t_rng) in t_rngs:
                _data_var[tlabel] = f.variables[varname_used][t_rng, lat_rng, lon_rng]

        _data[varname] = _data_var 


    data[sim_casename] = _data


lat_chop = lat[lat_rng]
lon_chop = lon[lon_rng]
mask_chop = mask[lat_rng, :][:, lon_rng]

mask_lnd = mask_chop == 1.0
#print("lat_chop: ", lat_chop)
eq_lat_rng = np.logical_and(lat_chop > -0.5, lat_chop < 0.5)

def computeCorrmap(idx, data, lat, lon):
            
    _idx_a = idx - np.mean(idx)
    _idx_std = np.std(_idx_a)

    corr = np.zeros((len(lat), len(lon)))
    for j in range(len(lat)):
        for i in range(len(lon)):
            _data = data[:, j, i]
            _data_a = _data - np.mean(_data, axis=0)
            corr[j, i] = np.sum(_idx_a * _data_a)/ ( np.std(_data_a) * _idx_std * (len(_idx_a)-1) )

    return corr

for sim_casename in sim_casenames:
    print("Post processing data of case %s" % sim_casename)        
    _data_TAUY_CXEQ = {}
    _data_TAUX_EQ = {}
    _data_SST_EQ = {}
    _data_corrmap_TAUY = {}
    _data_corrmap_TAUX = {}

    for (tlabel, t_rng) in t_rngs:
        _data_TAUY_CXEQ[tlabel] = np.mean(data[sim_casename]["TAUY"][tlabel][:, eq_lat_rng, :], axis=(1,2))
        _data_TAUX_EQ[tlabel] = np.mean(data[sim_casename]["TAUX"][tlabel][:, eq_lat_rng, :], axis=(1,2))
        _data_SST_EQ[tlabel] = np.mean(data[sim_casename]["SST"][tlabel][:, eq_lat_rng, :], axis=(1,2))
    
        # Map correlation
        _data_corrmap_TAUY[tlabel] = computeCorrmap(_data_TAUY_CXEQ[tlabel], data[sim_casename]["PRECC"][tlabel], lat_chop, lon_chop)
        _data_corrmap_TAUX[tlabel] = computeCorrmap(_data_TAUX_EQ[tlabel], data[sim_casename]["PRECC"][tlabel], lat_chop, lon_chop)
               

    print("Std of TAUY-cxeq JJA: ", np.std(_data_TAUY_CXEQ["JJA"]))
    print("Std of TAUY-cxeq DJF: ", np.std(_data_TAUY_CXEQ["DJF"]))
    print("Std of TAUX-eq JJA: ", np.std(_data_TAUX_EQ["JJA"]))
    print("Std of TAUX-eq DJF: ", np.std(_data_TAUX_EQ["DJF"]))
    print("Std of SST-eq JJA: ", np.std(_data_SST_EQ["JJA"]))
    print("Std of SST-eq DJF: ", np.std(_data_SST_EQ["DJF"]))
 
    data[sim_casename]["TAUY-cxeq"] = _data_TAUY_CXEQ
    data[sim_casename]["TAUX-eq"]   = _data_TAUX_EQ
    data[sim_casename]["corrmap-TAUY"]   = _data_corrmap_TAUY
    data[sim_casename]["corrmap-TAUX"]   = _data_corrmap_TAUX

    

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

fig = plt.figure(figsize=(5 * len(t_rngs), 3*len(sim_casenames)))
heights = [1] * len(sim_casenames)
widths  = [1,] * len(t_rngs) + [0.05]
spec = fig.add_gridspec(nrows=len(heights), ncols=len(widths), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3, right=0.8)

    
ax = []

for i, (tlabel, _) in enumerate(t_rngs):

    for j, sim_casename in enumerate(sim_casenames):

        row_idx = j

        print(sim_casename)

        _ax = fig.add_subplot(spec[row_idx, i], **proj_kw)
        ax.append(_ax)

        _TAUY = data[sim_casename]["TAUY-cxeq"][tlabel]
        _corrmap_TAUY = data[sim_casename]["corrmap-TAUY"][tlabel]
        _corrmap_TAUX = data[sim_casename]["corrmap-TAUX"][tlabel]
        print("_TAUY shape: ", _corrmap_TAUY.shape)
        
        _corrmap_TAUY[mask_lnd] = np.nan

        mappable = _ax.contourf(lon_chop, lat_chop, _corrmap_TAUY,  np.linspace(-0.8, 0.8, 9),  cmap=cm.get_cmap("bwr"), transform=data_proj, extend="both")
        _ax.coastlines()

        gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=1, color='gray', alpha=0.3, linestyle='-')
        
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlocator = mticker.FixedLocator([-60, -40, -20, 0, 20])
        gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        gl.xlabel_style = {'size': 8, 'color': 'black', 'ha':'center'}
        gl.ylabel_style = {'size': 8, 'color': 'black', 'ha':'right'}
        
        if i==0:
            _ax.text(-0.15, 0.5, "%s" % (sim_casename, ), fontsize=15, rotation=90, ha="right", va="center", transform=_ax.transAxes)
        
        if row_idx == 0:
            _ax.set_title(tlabel, size=15)
        


for _ax in ax:
    _ax.set_aspect('auto')
#    _ax.set_ylim([-90, 90])
#    _ax.set_yticks([])#[-90, -60, -30, 0, 30, 60, 90])
#    _ax.set_yticklabels([])
    #["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])

    _ax.set_xlim([120, 180])
#            _ax.set_xticks(np.linspace(0,360,7), crs=proj1)
#            _ax.set_xticklabels([])#["0", "60E", "120E", "180", "120W", "60W", "0"])


cax = fig.add_subplot(spec[0, -1])
cb = fig.colorbar(mappable,  cax=cax, ticks=np.linspace(-.8, .8, 5), orientation="vertical")
cb.set_label("Correlation")
#fig.savefig("%s/SSTA_correlation_col.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)

