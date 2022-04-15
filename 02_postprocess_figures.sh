#!/bin/bash


mkdir final_figures



convert figures/fig02_ctl-variability_SST_col.png -gravity North -chop x600 -gravity South -chop x600 -gravity West -chop 100x final_figures/fig02.png

convert figures/fig03_ctl-variability_SAT_col.png -gravity North -chop x500 -gravity South -chop x450 -gravity West -chop 300x -gravity East -chop 450x final_figures/fig03.png

convert figures/fig04_SSTA_correlation_col.png -gravity North -chop x500 -gravity South -chop x600 -gravity East -chop 600x -gravity West -chop 300x final_figures/fig04.png

cp figures/fig05_EOF_map_PDO.png final_figures/fig05.png

cp figures/fig06_spectrum_PDO.png final_figures/fig06.png

convert figures/fig07_diff_map_SST_col.png        -gravity North -chop x600 -gravity South -chop x600 -gravity West -chop 100x final_figures/fig07.png

cp figures/fig08_AMOC_psi.png final_figures/fig08.png


convert figures/fig09_CTL_zmean_cx_diff_EXP_MEAN_col.png -gravity North -chop x500 -gravity South -chop x200 -gravity West -chop 500x -gravity East -chop 500x final_figures/fig09.png

convert figures/fig10_diff_map_PREC_TOTAL_col.png -gravity North -chop x600 -gravity South -chop x600 -gravity West -chop 100x final_figures/fig10.png

cp figures/fig11_diff_zmean_PREC.png final_figures/fig11.png

convert figures/fig12_tropical_response_analysis.png -gravity North -chop x200 -gravity South -chop x200 -gravity West -chop 500x -gravity East -chop 650x final_figures/fig12.png

cp figures/fig13_tropical_response_analysis_EMOM_OHT.png final_figures/fig13.png

cp figures/fig14_heat_transport_analysis.png final_figures/fig14.png


cp figures/figS01_importance_of_KH.png final_figures/figS01.png

# Merging two sub-figures
convert \( figures/figS02a_ocean_mean_temperature_timeseries.png -background white -splice 0x300 -gravity NorthWest -pointsize 200 -annotate +200+200 '(a)' \) \
    \( figures/figS02b_AMOC_timeseries.png -background white -splice 0x300 -gravity NorthWest -pointsize 200 -annotate +200+200 '(b)' \) -gravity center -append \
     final_figures/figS02.png

cp figures/figS03_OHC_trend.png final_figures/figS03.png
convert figures/figS04_CTL_map_diff_MEAN_col.png -gravity North -chop x500 -gravity South -chop x500 -gravity West -chop 100x final_figures/figS04.png
cp figures/figS05_CTL_zmean_precip.png final_figures/figS05.png
cp figures/figS06_seaice_target_map_contour.png final_figures/figS06.png
convert figures/figS06_seaice_target_map_contour.png -gravity West -chop 2000x -gravity East -chop 3000x final_figures/figS06.png

#convert tropical_response_analysis_OHC.png -gravity North -chop x600 -gravity South -chop x600 -gravity West -chop 100x tropical_response_analysis_OHC_crop.png


