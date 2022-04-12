import numpy as np

def ext(data):
    s = data.shape
    ndata = np.zeros((s[0], s[1]+1))
    ndata[:, 0:-1] = data
    ndata[:, -1] = data[:, 0]
    return ndata
 
def ext_axis(lon):
    return np.append(lon, 360) 


def getSimVars(varnames):
    all_sim_vars = {
        "CORR"       : "atm_analysis_SST_CORR.nc",
        "AHT"        : "atm_analysis_AHT_OHT.nc",
        "OHT"        : "ocn_analysis_OHT.nc",
        "OHT_WKRSTT"        : "ocn_analysis_OHT.nc",
        "OHT_ADVT"        : "ocn_analysis_OHT.nc",
        "WKRSTT_avg"        : "ocn_analysis_OHT.nc",
        "aice"       : "ice_analysis_mean_anomaly_aice.nc",
        "vice"       : "ice_analysis_mean_anomaly_vice.nc",
        "SST"        : "atm_analysis_mean_anomaly_SST.nc",
        "PREC_TOTAL" : "atm_analysis_mean_anomaly_PREC_TOTAL.nc",
        "PSL"        : "atm_analysis_mean_anomaly_PSL.nc",
        "STRAT"      : "ocn_analysis_mean_anomaly_STRAT.nc",
        "TREFHT"     : "atm_analysis_mean_anomaly_TREFHT.nc",
        "FSNT"     : "atm_analysis_mean_anomaly_FSNT.nc",
        "FLNT"     : "atm_analysis_mean_anomaly_FLNT.nc",
        "LHFLX"     : "atm_analysis_mean_anomaly_LHFLX.nc",
        "SHFLX"     : "atm_analysis_mean_anomaly_SHFLX.nc",
        "TAUX"      : "atm_analysis_mean_anomaly_TAUX.nc",
        "TAUY"      : "atm_analysis_mean_anomaly_TAUY.nc",
        "T"      : "atm_analysis_mean_anomaly_T_zm.nc",
        "U"      : "atm_analysis_mean_anomaly_U_zm.nc",
        "Z3"      : "atm_analysis_mean_anomaly_Z3_zm.nc",
        "ice_volume_NH"     : "ice_analysis_total_seaice.nc",
        "ice_volume_SH"         : "ice_analysis_total_seaice.nc",
        "ice_volume_GLB"        : "ice_analysis_total_seaice.nc",
        "ice_area_NH"         : "ice_analysis_total_seaice.nc",
        "ice_area_SH"         : "ice_analysis_total_seaice.nc",
        "ice_area_GLB"        : "ice_analysis_total_seaice.nc",
        "ice_extent_NH"         : "ice_analysis_total_seaice.nc",
        "ice_extent_SH"         : "ice_analysis_total_seaice.nc",

    }


    r = {}
    for varname in varnames:
        r[varname] = all_sim_vars[varname]

    return r


def getSimcases(casenames):

    all_simcases = {

        "SOM" : {
            "CTL": "CTL_21-120/SOM_CTL_coupled",
            "EXP": "EXP_81-180/SOM_EXP_coupled",
            "lc" : "red",
            "ls" : "-",
            "ax_idx" : (0, 0),
            "col_idx" : 3,
        },

        "MLM" : {
            "CTL": "CTL_21-120/MLM_CTL_coupled",
            "EXP": "EXP_81-180/MLM_EXP_coupled",
            "lc" : "limegreen",
            "ls" : "-",
            "ax_idx" : (0, 1),
            "col_idx" : 2,
        },
        
        "EMOM" : {
            "CTL": "CTL_21-120/EMOM_CTL_coupled",
            "EXP": "EXP_81-180/EMOM_EXP_coupled",
            "lc" : "dodgerblue",
            "ls" : "-",
            "ax_idx" : (0, 2),
            "col_idx" : 1,
        },

        "OGCM" : {
            "CTL": "CTL_21-120/POP2_CTL",
            "EXP": "EXP_POP2/POP2_EXP_81-180",
            "lc" : "black",
            "ls" : "-",
            "ax_idx" : (0, 3),
            "col_idx" : 0,
        },

        "OGCM_151-250" : {
            "CTL": "CTL_21-120/POP2_CTL",
            "EXP": "EXP_POP2/POP2_EXP_151-250",
            "lc" : "black",
            "ls" : "-",
            "ax_idx" : (0, 3),
        },

        "OGCM_251-350" : {
            "CTL": "CTL_21-120/POP2_CTL",
            "EXP": "EXP_POP2/POP2_EXP_251-350",
            "lc" : "black",
            "ls" : "-",
            "ax_idx" : (0, 3),
        },

        "OGCM_351-450" : {
            "CTL": "CTL_21-120/POP2_CTL",
            "EXP": "EXP_POP2/POP2_EXP_351-450",
            "lc" : "black",
            "ls" : "-",
            "ax_idx" : (0, 3),
        },

        "MLM_tau01" : {
            "CTL": "CTL_21-120/MLM_CTL_coupled",
            "EXP": "SRT_21-120/MLM_tau01year",
            "lc" : "limegreen",
            "ls" : "-",
            "ax_idx" : (0, 1),
            "col_idx" : 2,
        },
 

    }

    r = {}
    for casename in casenames:
        r[casename] = all_simcases[casename]


    return r
