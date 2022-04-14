import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib as mplt
import matplotlib.ticker as mticker
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



loaded_ocn_models = ["MLM", "EMOM", "POP2" ]

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
omega = 2 * np.pi / 86400.0
f_coriolis = 2 * omega * np.sin(llat * np.pi/180)
epsilon = 1.41e-5
f2_ep2 = f_coriolis ** 2.0 + epsilon ** 2.0
rho = 1026.0
c_p = 3996.0
H_EK = 50.0
#dx = 2 * np.pi * np.cos(np.pi/180 * lat) / len(lon)

data = {}

for ocn_model in loaded_ocn_models:

    filename = "data/supp/OHC_data/OHC_diff_%s.nc" % (ocn_model,)
        
    with Dataset(filename, "r") as f:

        print("Loading file: %s" % (filename,) )

        if ocn_model == "POP2":
            z_t = f.variables["z_t"][:] / 100
            z_w_top = f.variables["z_w_top"][:] / 100
            z_w_bot = f.variables["z_w_bot"][:] / 100
            dz = z_w_bot - z_w_top
            
        else:
            dz = f.variables["dz_cT"][:, 0, 0]

        dz = dz[0:33]
        
        TEMP = f.variables["TEMP"][0, 0:33, :, :]
        TEMP[np.isnan(TEMP)] = 0.0
        data[ocn_model] = np.nansum(TEMP * dz[:, None, None], axis=0) / 365 / 86400 / 100 / sum(dz)
    
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
# OHC analysis
ocn_models = ["POP2", "EMOM", "MLM"]

heights = [1] * len(ocn_models)
widths  = [1, 0.05]
fig = plt.figure(figsize=(5, 3*len(ocn_models)))
spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3, right=0.8) 

ax = []
for i, ocn_model in enumerate(ocn_models):
    
    print("Plotting ", ocn_model)

    _ax = fig.add_subplot(spec[i, 0], **proj_kw)
    ax.append(_ax)
   
    _ax.set_title("SIL_%s" % ocn_model) 
       

    if ocn_model == "POP2":
        _ax.set_title("SIL_OGCM")
 
    OHC = data[ocn_model] * 100 * 86400 * 365
    cmap_OHC = cm.get_cmap("bwr")
    clev_OHC =      np.linspace(-1, 1, 11)
    clevticks_OHC = np.linspace(-1, 1, 11)
    mappable = _ax.contourf(lon, lat, OHC,  clev_OHC,  cmap=cmap_OHC, transform=data_proj, extend="both")

    if i == 0:
        cax = fig.add_subplot(spec[0, -1])
        cb = fig.colorbar(mappable,  cax=cax, ticks=clevticks_OHC, orientation="vertical")
        cb.set_label("Drift rate [ $ {}^\\circ\\mathrm{C}\\,/\\, 100\\mathrm{yr}$ ] ")

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

fig.savefig("%s/figS03_OHC_trend.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)

