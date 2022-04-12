include("SpectrumTools.jl")

using .SpectrumTools
using NCDatasets
using Plots
using PyCall
using JSON
using ArgMacros


args = @dictarguments begin
    @argumentrequired String EOF_name "--EOF"
end


EOF_name = args[:EOF_name]# "NAO"

plot_cases = Dict(
    "ENSO"   => [ "OGCM", "EMOM", "MLM", "SOM"], 
    "AO"   => ["OGCM", "EMOM", "MLM", "SOM"], 
    "AAO"  => ["OGCM", "EMOM", "MLM", "SOM"], 
    "NAO"  => ["OGCM", "EMOM", "MLM", "SOM"], 
    "PDO"  => ["OGCM", "EMOM", "MLM", "SOM"], 
)

s = "CTL"

casenames = plot_cases[EOF_name]

println("Getting cases...")
pushfirst!(PyVector(pyimport("sys")."path"), ".")
tools = pyimport("quick_tools")
simcases = tools.getSimcases(casenames)
println("Done")

line_cfg = Dict(
    "OGCM" => ["black", "-"],
    "EMOM" => ["dodgerblue", "--"],
    "MLM"  => ["orangered", "-."],
    "SOM"  => ["green", ":"],
)



pick_mode = Dict(

    "ENSO" => Dict(
        "OGCM" => 1,
        "EMOM" => 1,
        "MLM"  => 1,
        "SOM"  => 1,
    ),

    "AO" => Dict(
        "OGCM" => 1,
        "EMOM" => 1,
        "MLM"  => 1,
        "SOM"  => 1,
    ),

    "AAO" => Dict(
        "OGCM" => 1,
        "EMOM" => 1,
        "MLM"  => 1,
        "SOM"  => 1,
    ),

    "NAO" => Dict(
        "OGCM" => 1,
        "EMOM" => 1,
        "MLM"  => 1,
        "SOM"  => 1,
    ),

    "PDO" => Dict(
        "OGCM" => 1,
        "EMOM" => 1,
        "MLM"  => 1,
        "SOM"  => 1,
    ),


)



JSON.print(simcases, 4)

println("Doing EOF : $EOF_name")

data = Dict()

for scenario in ["CTL",]
#for scenario in ["CTL", "EXP"]

    local _tmp = Dict()
    for (casename, caseinfo) in simcases
        println("Loading $casename")
        Dataset("data/$(caseinfo[scenario])/atm_analysis_$(EOF_name).nc", "r") do ds
            _tmp[casename] = nomissing(ds["PCAs_ts"][pick_mode[EOF_name][casename], :], NaN)
        end
        data[scenario] = _tmp
    end


end

println("Loading PyPlot")
using PyPlot
plt = PyPlot
println("done")

fig, ax = plt.subplots(1, 1)

fig.suptitle("$s $EOF_name")
for (casename, caseinfo) in simcases

    ts = data[s][casename]

    months = length(ts)
    years  = Int64(months / 12)


    hspec = hfft(ts)
    wavenumber = collect(Float64, 0:(length(hspec)-1))
    hfreq = wavenumber / years

    marked_periods = [1, 3, 4, 5, 7, 10, 20]
    ticks = marked_periods.^(-1)

    linecolor, linestyle = line_cfg[casename]

#    bpf1 = genBandpassFilter(hfreq, 1.0/10.0,  1.0/2.0, 1/20000)
#    bpf2 = genBandpassFilter(hfreq, 1.0/10.0,  1.0/2.0, 1/10)

#    filtered1_hspec = hspec .* bpf1
#    filtered2_hspec = hspec .* bpf2

#    filtered1_NINO34 = hifft(filtered1_hspec)
#    filtered2_NINO34 = hifft(filtered2_hspec)


    strength = abs.(hspec)
    mavg_strength = mavg(strength, 1)

    ax.scatter(hfreq, strength, s=10, marker="o",)
    ax.plot(hfreq, mavg_strength, color=linecolor, linestyle="-", label="$casename")
    ax.set_xticks(ticks)
    ax.set_xticklabels(marked_periods)

end

ax.legend()
ax.set_xlim([0, 1])

plt.show()

fig.savefig("graph/EOF_spectrum_$(EOF_name)_$(s).png", dpi=200)






