include("SpectralAnalysis.jl")

using .SpectralAnalysis

using NCDatasets
using Statistics
using ArgMacros
using PyCall


function detrend(arr::AbstractArray)
    N = length(arr)
    t = collect(Float64, 1:N)
    A = zeros(Float64, N, 2)
    A[:, 1] .= 1.0
    A[:, 2] .= t
    arr_d = arr - A * ( (A' * A ) \ ( A' * arr ) )
end


args = @dictarguments begin
    @argumentrequired String EOF_name "--EOF"
end

using JSON
JSON.print(args, 4)

# setup

EOF_name = args[:EOF_name]
casenames = ["SOM", "MLM", "EMOM", "OGCM",][end:-1:1]

println("Plotting the EOF : $EOF_name")

println("Getting cases...")
pushfirst!(PyVector(pyimport("sys")."path"), ".")
tools = pyimport("quick_tools")
simcases = tools.getSimcases(casenames)
println("Done")


t_idx = [1,]

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
)[EOF_name]



data = Dict()
years = nothing
    
for (casename, caseinfo) in simcases
 
    caseinfo = caseinfo["CTL"]
    filename = "data/hierarchy_statistics/$(caseinfo)/atm_analysis_$(EOF_name).nc"
    println("Loading $casename : $filename")

    Dataset(filename, "r") do ds
        
        global years
       
        #_tmp[casename] = nomissing(ds["PCAs_ts"][pick_mode[casename], :], NaN)
        idx = nomissing(ds["PCAs_ts"][pick_mode[casename], :], NaN)
        idx = mean(reshape(idx, 12, :)[t_idx, :], dims=1)[1, :]
        #if years == nothing
            years = length(idx)
            println("Years : $years")
        #end

        idx = detrend(idx)
        data[casename] = Dict( "idx" => idx )
 
    end
end

    
lag_window_M = floor(Int64, years / 4)
marked_periods = [80, 50, 20, 10, 5, 3]
ticks = years ./ marked_periods

println("Loading PyPlot...")
using PyPlot
plt = PyPlot
println("Done.")
fig, ax = plt.subplots(length(casenames), 1, figsize=(10, 3 * length(casenames)), gridspec_kw = Dict("hspace" => 0.3), constrained_layout=true)


new_ylim = [0.0, 0.0]

for (i, casename) in enumerate(casenames)

    println("Doing casename: $casename")
    idx = data[casename]["idx"]

    println("Mean of idx: $(mean(idx))")

    spec_none,   _, _       = computeSpectrum(idx; smoothing="none")
    spec_tukey, dω, λ_tukey = computeSpectrum(idx; smoothing="Tukey", lag_window_M = lag_window_M)

    white_noise = transpose(spec_none) * dω / sum(dω)
    wavenumber = collect(Float64, 1:length(spec_tukey))
    lowerbnd, upperbnd = computeCIRatio(length(idx), λ_tukey; α=0.05)

    ax[i].fill_between(wavenumber, spec_tukey * upperbnd, spec_tukey * lowerbnd, color="orange", alpha=0.2)
    #ax[i].plot(wavenumber, spec_none, color="gray",   linestyle="dashed")
    ax[i].plot(wavenumber, spec_tukey, color="black", linestyle="solid")
    ax[i].plot([wavenumber[1], wavenumber[end]], [1.0, 1.0] * white_noise, "r--",)
    
    ax[i].set_title(casename)

    ax[i].set_xticks(ticks)
    ax[i].set_xticklabels([ "$s" for s in marked_periods])
    ax[i].set_xlim([ticks[1], ticks[end]])
    ax[i].set_xlabel("Period [yr]")
    ax[i].set_ylabel("\$ \\left|\\hat{f}(\\omega)\\right|^2 \$")

    println(ax[i].get_ylim())
    new_ylim[1] = min(new_ylim..., ax[i].get_ylim()...)    
    new_ylim[2] = max(new_ylim..., ax[i].get_ylim()...)    

end

for (i, casename) in enumerate(casenames)
    ax[i].set_ylim(new_ylim)
end

fig.suptitle("Periodiogram of $(EOF_name) (Tukey lag window M = $(lag_window_M))")

fig.savefig("figures/fig06_spectrum_$(EOF_name).png", dpi=300)
plt.show()


#=
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for (i, casename) in enumerate(casenames)

    idx = data[casename]["idx"]

    ax.plot(idx, label=casename)
end
ax.legend()

plt.show()
=#




