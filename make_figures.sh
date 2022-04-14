#!/bin/bash

py=python3
jl=julia


plot_codes=(
    $py plot_fig02.py
    $py plot_fig03.py
    $py plot_fig04.py
    $py plot_fig05.py
    $jl "plot_fig06.jl --EOF PDO"
    $py plot_fig07.py
    $py plot_fig08.py
    $py plot_fig09.py
    $py plot_fig10.py
    $py plot_fig11.py
    $py plot_fig12_fig13.py
    $py plot_fig14.py
    $py plot_figS01.py
    $py plot_figS02b.py
    $py plot_figS03.py
    $py plot_figS04.py
    $py plot_figS05.py
    $py plot_figS06.py
)

# Some code to download data and extract them



mkdir figures
mkdir data_extra

echo "Making extra data into data_extra"
$jl remove_ENSO_annual.jl


N=$(( ${#plot_codes[@]}  ))
echo "We have $N file(s) to run..."
for i in $( seq 1 $(( ${#plot_codes[@]} / 2 )) ) ; do
    PROG="${plot_codes[$(( (i-1) * 2 + 0 ))]}"
    FILE="${plot_codes[$(( (i-1) * 2 + 1 ))]}"
    echo "=====[ Running file: $FILE ]====="
    eval "$PROG $FILE" &
done


wait
