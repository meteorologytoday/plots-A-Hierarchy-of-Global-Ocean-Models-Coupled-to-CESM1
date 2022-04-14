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
output_dir = "graph"



loaded_ocn_models = ["SOM", "MLM", "EMOM", "POP2" ]
#ocn_models = ["EMOM", "POP2" ]
#ocn_models = ["EMOM", ]

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

for scenario in ["CTL", "EXP"]:

    data[scenario] = {}

    for ocn_model in loaded_ocn_models:

        casename = "%s_%s" % (ocn_model, scenario)
        _tmp = {}
        
        atm_filename = "data/raw_averaged_result/%s/atm.nc" % (casename,)
        ocn_filename = "data/raw_averaged_result/%s/ocn_regrid.nc" % (casename, )
            
        with Dataset(atm_filename, "r") as f:
            print("Loading file: %s" % (atm_filename,) )
            for varname in ["PRECC", "PRECL", "TAUX", "TAUY", "SST", "FLNS", "FSNS", "LHFLX", "SHFLX", "SWCF"]: 
                _tmp[varname] = f.variables[varname][0, :, :]
                
            _tmp["Usfc"] = f.variables["U"][0,-1, :, :]
            _tmp["Vsfc"] = f.variables["V"][0,-1, :, :]
 
        with Dataset(ocn_filename, "r") as f:
            print("Loading file: %s" % (ocn_filename,) )
            #_tmp["UVEL"] = f.variables["UVEL"][0, 0, :, :]
            #_tmp["VVEL"] = f.variables["VVEL"][0, 0, :, :]
            _tmp["WVEL"] = f.variables["WVEL"][0, 5, :, :]
            #_tmp["TAUX_east"]  = f.variables["TAUX"][0, 0, :, :]
            #_tmp["TAUY_north"] = f.variables["TAUY"][0, 0, :, :]
                
            _tmp["TEMP"] = f.variables["TEMP"][0, :, :, :]


            if ocn_model == "POP2":
                z_t = f.variables["z_t"][:] / 100
                z_w_top = f.variables["z_w_top"][:] / 100
                z_w_bot = f.variables["z_w_bot"][:] / 100
                dz = z_w_bot - z_w_top
                
                _tmp["WVEL"] /= 100.0
            else:
                dz = f.variables["dz_cT"][:, 0, 0]
                #_tmp["ADVT"] = f.variables["ADVT"][0, :, :, :]
                #_tmp["ADVT_TOT"] = np.sum(_tmp["ADVT"][0:33, :, :] * dz[0:33, None, None], axis=0)
                #print("Max of ADVT_TOT: ", np.amax(_tmp["ADVT_TOT"]))

                

            T1  = np.average(_tmp["TEMP"][0:5,  :, :], axis=0, weights=dz[0:5])
            T2  = np.average(_tmp["TEMP"][5:33, :, :], axis=0, weights=dz[5:33])
            _tmp["OHC"] = np.sum(_tmp["TEMP"][:, :, :] * dz[:, None, None], axis=0) * rho * c_p


            _tmp["DELTA_TEMP"] = T1 - T2 
           
        _tmp["NHFLX"] = - _tmp["FSNS"] + _tmp["FLNS"] + _tmp["SHFLX"] + _tmp["LHFLX"]
        _tmp["PREC_TOTAL"] = _tmp["PRECC"] + _tmp["PRECL"]
        _tmp["SST"] -= 273.15
        _tmp["SST"][lnd_mask_idx] = np.nan
        
        _tmp["TAUX"][lnd_mask_idx] = np.nan
        _tmp["TAUY"][lnd_mask_idx] = np.nan
       
        _tmp["TAUX"] *= -1
        _tmp["TAUY"] *= -1


        _tmp["UVEL_EK"] = (  f_coriolis * _tmp["TAUY"] + epsilon * _tmp["TAUX"]) / f2_ep2 / rho / H_EK
        _tmp["VVEL_EK"] = (- f_coriolis * _tmp["TAUX"] + epsilon * _tmp["TAUY"]) / f2_ep2 / rho / H_EK




        data[scenario][ocn_model] = _tmp

def genDIF(om, v):
    return data["EXP"][om][v] - data["CTL"][om][v]

def genCTL(om, v):
    return data["CTL"][om][v]


# This section I just wanna compute the mean difference of tropical
# precipitation numerically.

valid_idx = (np.abs(llat) > 30.0)
    
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
ocn_models = ["POP2", "EMOM", "MLM", "SOM"]

heights = [1] * len(ocn_models)
widths  = [1, 0.05]
fig = plt.figure(figsize=(5, 3*len(ocn_models)))
spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3, right=0.8) 

ax = []
for i, ocn_model in enumerate(ocn_models):
    
    print("Plotting ", ocn_model)

    _ax = fig.add_subplot(spec[i, 0], **proj_kw)
    ax.append(_ax)
   
    _ax.set_title("RESP_%s" % ocn_model) 
       

    if ocn_model == "POP2":
        _ax.set_title("RESP_OGCM")
 
    OHC_DIF = genDIF(ocn_model, "OHC") / 1e9
    print("OHC max: ", np.amax(np.abs(OHC_DIF)))   
    cmap_OHC = cm.get_cmap("bwr")
    clev_OHC =      np.linspace(-5, 5, 21)
    clevticks_OHC = np.linspace(-5, 5, 11)
    mappable = _ax.contourf(lon, lat, OHC_DIF,  clev_OHC,  cmap=cmap_OHC, transform=data_proj, extend="both")

    if i == 0:
        cax = fig.add_subplot(spec[0, -1])
        cb = fig.colorbar(mappable,  cax=cax, ticks=clevticks_OHC, orientation="vertical")
        cb.set_label("$\\Delta \\mathrm{OHC}$ [ $ \\times 10^{9} \\, \\mathrm{J}$ ] ")

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

fig.savefig("%s/tropical_response_analysis_OHC.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)


#######################################
# Wind vector , SST, upwelling analysis
ocn_models = ["POP2", "EMOM"]

heights = [1] * len(ocn_models)
widths  = [1, 0.05]
fig = plt.figure(figsize=(12, 3*len(ocn_models)))
spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3, right=0.8) 


ax = []
def genDIF(om, v):
    return data["EXP"][om][v] - data["CTL"][om][v]

def genCTL(om, v):
    return data["CTL"][om][v]


for i, ocn_model in enumerate(ocn_models):
    
 
    print("Plotting ", ocn_model)

    _ax = fig.add_subplot(spec[i, 0], **proj_kw)
    ax.append(_ax)
   
    _ax.set_title("RESP_%s" % ocn_model) 
       

    if ocn_model == "POP2":
        _ax.set_title("RESP_OGCM")
 
    WVEL_DIF = genDIF(ocn_model, "WVEL") * 86400.0 * 100
    #UVEL_DIF = genDIF(ocn_model, "UVEL") * 86400.0 * 100
    #VVEL_DIF = genDIF(ocn_model, "VVEL") * 86400.0 * 100
    SST_DIF = genDIF(ocn_model, "SST")
    TAUX_DIF = genDIF(ocn_model, "TAUX") * 1e2
    TAUY_DIF = genDIF(ocn_model, "TAUY") * 1e2
    
    Usfc_DIF = genDIF(ocn_model, "Usfc")
    Vsfc_DIF = genDIF(ocn_model, "Vsfc")
    
    NHFLX_DIF = genDIF(ocn_model, "SWCF")
 
    #TAUX_east_DIF  = genDIF(ocn_model, "TAUX_east")  * 1e2
    #TAUY_north_DIF = genDIF(ocn_model, "TAUY_north") * 1e2
    
    UVEL_EK_DIF = genDIF(ocn_model, "UVEL_EK") * 1e2
    VVEL_EK_DIF = genDIF(ocn_model, "VVEL_EK") * 1e2
    
    PREC_TOTAL_DIF = genDIF(ocn_model, "PREC_TOTAL") * 86400 * 365 * 1000
        
    cmap_SST = cm.get_cmap("bwr")
    clev_SST = np.linspace(-0.3, 0.3, 13)
    clevticks_SST = np.linspace(-0.3, 0.3, 7)
    mappable = _ax.contourf(lon, lat, SST_DIF,  clev_SST,  cmap=cmap_SST, transform=data_proj)#, extend="both")

    if i == 0:
        cax = fig.add_subplot(spec[0, -1])
        cb = fig.colorbar(mappable,  cax=cax, ticks=clevticks_SST, orientation="vertical")
        cb.set_label("$ \Delta $SST [ ${}^\\circ \\mathrm{C}$ ] ")

    print("max w: ", np.amax(WVEL_DIF))
    print("min w: ", np.amin(WVEL_DIF))
    clev_W = np.array([-15, -10, -5, -1, 1, 5, 10, 15]) #np.linspace(-1, 1, 21) * 20
    CS = _ax.contour(lon, lat, WVEL_DIF,  clev_W, transform=data_proj, colors="black", linewidths=1)
    _ax.clabel(CS, CS.levels, inline=True, inline_spacing=0, fmt="%d", fontsize=10) 
    
    if ocn_model == "POP2":
        #CS = _ax.contour(lon, lat, NHFLX_DIF, [-0.8, 0.8], transform=data_proj, colors="yellow", linewidths=2)
    #if ocn_model == "POP2":
        CS = _ax.contourf(lon, lat, NHFLX_DIF, [-1000, -1, 1, 1000], colors="none", hatches=["//", None, ".."], transform=data_proj)
        for i, collection in enumerate(CS.collections):
            collection.set_edgecolor("yellow")
            collection.set_linewidth(0.)
 
    #print("max TAUX: ", np.abs(TAUX_DIF))
    #print("min TAUX: ", np.amin(WVEL_DIF))
    print("max abs UVEL_EK: ", np.amax(np.abs(UVEL_EK_DIF)))
    print("max abs VVEL_EK: ", np.amax(np.abs(VVEL_EK_DIF)))
      
    skip_x = 5
    skip_y = 2
    #_ax.quiver(lon[::skip_x], lat[::skip_y], TAUX_DIF[::skip_y, ::skip_x], TAUY_DIF[::skip_y, ::skip_x], pivot="tail", color="black", scale_units="inches", scale=4, transform=data_proj)

    #_ax.quiver(lon[::skip_x], lat[::skip_y], TAUX_east_DIF[::skip_y, ::skip_x], TAUY_north_DIF[::skip_y, ::skip_x], pivot="tail", color="darkred", scale_units="inches", scale=2, transform=data_proj)
    qv_ek   = _ax.quiver(lon[::skip_x], lat[::skip_y], UVEL_EK_DIF[::skip_y, ::skip_x], VVEL_EK_DIF[::skip_y, ::skip_x], pivot="tail", color="lime", scale_units="inches", scale=3, transform=data_proj)

    qv_wind = _ax.quiver(lon[::skip_x], lat[::skip_y], Usfc_DIF[::skip_y, ::skip_x], Vsfc_DIF[::skip_y, ::skip_x], pivot="tail", color="black", scale_units="inches", scale=3, transform=data_proj)

    qvk_ek   = plt.quiverkey(qv_ek,   X=0.82, Y=0.2,  U=1,    label="$ \\vec{v}_{\\mathrm{EK}}$ $ 1 \\, \\mathrm{cm} / \\mathrm{s}$", labelcolor="black", coordinates="figure")
    qvk_wind = plt.quiverkey(qv_wind, X=0.82, Y=0.3,  U=1,  label="Surface wind $ 1 \\, \\mathrm{m} / \\mathrm{s}$", labelcolor="black", coordinates="figure")
    
    qvk_wind.text.set_backgroundcolor('w')

    _ax.contourf(lon, lat, PREC_TOTAL_DIF, [-5000, -100, 100, 5000], alpha=0, hatches=['//', None, '..'], transform=data_proj )

for _ax in ax:
    
    _ax.coastlines()
    _ax.add_feature(cfeature.LAND, color="#cccccc")
    _ax.set_aspect('auto')


    gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.3, linestyle='-')
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([150, 180, -150, -120, -90, -60, -30])
    gl.ylocator = mticker.FixedLocator([-10, -5, 0, 5, 10])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {'size': 8, 'color': 'black', 'ha':'center'}
    gl.ylabel_style = {'size': 8, 'color': 'black', 'ha':'right'}
    
    #_ax.set_extent([100, 270, -30, 30], crs=ccrs.PlateCarree())
    _ax.set_extent([130, 350, -12, 12], crs=ccrs.PlateCarree())

fig.savefig("%s/tropical_response_analysis.png" % (output_dir,), dpi=600)
plt.show()
plt.close(fig)

#######################################
# EMOM tropical heat transport in TAUX and TAUY components    
DELTA_TEMP_DIF = genDIF("EMOM", "DELTA_TEMP")
DELTA_TEMP_CTL = genCTL("EMOM", "DELTA_TEMP")
TAUY_DIF = genDIF("EMOM", "TAUY")
TAUX_DIF = genDIF("EMOM", "TAUX")
TAUY_CTL = genCTL("EMOM", "TAUY")
TAUX_CTL = genCTL("EMOM", "TAUX")

#OHT_ADVT = np.sum(rho * c_p * genDIF("EMOM", "ADVT_TOT") * dx[:, None], axis=1) / 1e15
OHT_TAUY = np.sum(  c_p * DELTA_TEMP_CTL * epsilon    * TAUY_DIF / f2_ep2 * dx[:, None], axis=1) / 1e15
OHT_TAUX = np.sum(- c_p * DELTA_TEMP_CTL * f_coriolis * TAUX_DIF / f2_ep2 * dx[:, None], axis=1) / 1e15
OHT_DELTA_TEMP = np.sum(c_p * DELTA_TEMP_DIF * ( - f_coriolis * TAUX_CTL + epsilon * TAUY_CTL ) / f2_ep2 * dx[:, None], axis=1) / 1e15



# Load total OHT
with Dataset("data/CTL_21-120/EMOM_CTL_coupled/ocn_analysis_OHT.nc", "r") as f_CTL:
    with Dataset("data/EXP_81-180/EMOM_EXP_coupled/ocn_analysis_OHT.nc", "r") as f_EXP:
        OHT_ADVT = ( f_EXP.variables["OHT_ADVT_MEAN"][:] - f_CTL.variables["OHT_ADVT_MEAN"][:] ) / 1e15
        lat_bnd = f_EXP.variables["lat_bnd"][:]
        OHT_ADVT = np.interp(lat, lat_bnd, OHT_ADVT)



fig, ax = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True, sharex=True, gridspec_kw={'height_ratios': [1.5, 3]})

ax[0].set_title("(a) Change of wind stress of EMOM")
ax[1].set_title("(b) OHT decomposition of EMOM")

TAUX_DIF_MASKED = np.mean(np.ma.masked_where(lnd_mask_idx, TAUX_DIF), axis=1) * 1e3
TAUY_DIF_MASKED = np.mean(np.ma.masked_where(lnd_mask_idx, TAUY_DIF), axis=1) * 1e3

ax[0].plot(lat, TAUX_DIF_MASKED, color="black", linestyle="solid", label="$\\tau^x$")
ax[0].plot(lat, TAUY_DIF_MASKED, color="black", linestyle="dashed", label="$\\tau^y$")
ax[0].set_ylabel("Wind stress [ $ \\times 10^{-3} \\, \\mathrm{N} / \\, \\mathrm{m}^2$ ] ")

ax[1].plot(lat, OHT_ADVT, color="red", linestyle="solid", label="Total")
ax[1].plot(lat, OHT_TAUX, color="black", linestyle="solid", label="Due to $\\tau^x $")
ax[1].plot(lat, OHT_TAUY, color="black", linestyle="dashed", label="Due to $\\tau^y $")
ax[1].plot(lat, OHT_DELTA_TEMP, color="blue", linestyle="solid", label="Due to $\\Delta T$")
ax[1].plot(lat, OHT_ADVT - (OHT_TAUX + OHT_TAUY + OHT_DELTA_TEMP), color="green", linestyle="solid", label="Due to diffusion")

ax[1].set_ylabel("Heat transport [ PW ] ")
ax[1].set_xlabel("Latitude [ ${}^\circ \\mathrm{N}$ ] ")
ax[1].set_xlim([-30, 30])

ax[0].set_ylim([-3, 3])

ax[0].legend()
ax[1].legend()
ax[0].grid()
ax[1].grid()

fig.savefig("%s/tropical_response_analysis_EMOM_OHT.png" % (output_dir,), dpi=600)
plt.show()
###############
# vertical-longitudinal temperature change
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

valid_idx = (np.abs(lat) < 5.0)
    
TEMP_DIF = genDIF("POP2", "TEMP")[:, valid_idx, :]
TEMP_DIF = TEMP_DIF.filled(np.nan).mean(axis=1)

topo = TEMP_DIF * 1

topo_idx = np.isnan(topo)
topo[np.isfinite(topo)] = np.nan
topo[topo_idx] = 0.5
#topo[TEMP_DIF.mask == 0] = np.nan
print(topo)

#CS = ax.contour(lon, z_t, TEMP_DIF, 20, linewidths=1, colors="black")
#ax.clabel(CS, CS.levels, inline=True, inline_spacing=0, fmt="%.2f", fontsize=10)
ax.contourf(lon, z_t, topo, [0, 1, 2], colors=["#cccccc"])

cmap = cm.get_cmap("bwr")
clevticks = np.linspace(-0.5, 0.5, 7)
clevs     = np.linspace(-0.5, 0.5, 21)
mappable = ax.contourf(lon, z_t, TEMP_DIF, clevs, cmap=cmap, extend="both")
cb = fig.colorbar(mappable,  ax=ax, ticks=clevticks, orientation="vertical")
cb.set_label("$ \Delta $T [ ${}^\\circ \\mathrm{C}$ ] ")


ax.set_ylim([0, 2000])

ax.invert_yaxis()

plt.show()
 

# Compute temperature stratification


