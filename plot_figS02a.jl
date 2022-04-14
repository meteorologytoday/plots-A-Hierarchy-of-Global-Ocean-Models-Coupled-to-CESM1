using NCDatasets
using StatsBase
using DataStructures

function cyclic_mean(d :: AbstractArray{T, 1}, cnt :: Int64 = 12) where T <: Union{Float64, Float32}
    return mean(reshape(d, cnt, :), dims=1)[1, :]
end


data_dir = "data/supp/ocean_mean_temp"

cases = OrderedDict(
    "SIL_OGCM" => "paper2021_POP2_EXP",
    "SIL_EMOM" => "paper2021_EMOM_EXP",
    "SIL_MLM"  => "paper2021_MLM_EXP",
    "SIL_SOM"  => "paper2021_SOM_EXP",
)


data = Dict()

# loading data

for (label, casename) in cases
    
    _d = Dict()

    ds = Dataset("$data_dir/$(casename).ocn_mean_T.nc", "r")

    _d["TEMP"] = nomissing(ds["TEMP"][:], NaN)
    _d["SALT"] = nomissing(ds["SALT"][:], NaN)

    data[label] = _d
    close(ds)

    
end

Dataset("CESM_domains/ocn_zdomain.nc", "r") do ds

    global z_w_top = nomissing(ds["z_w_top"][:], NaN) / 100
    global z_w_bot = nomissing(ds["z_w_bot"][:], NaN) / 100

    global Δz_cT = z_w_bot - z_w_top

#    println(Δz_cT)
end


println("Loading PyPlot")
using PyPlot
println("done.")
plt = PyPlot

fig, ax = plt.subplots(1, 1, figsize=(6,4), constrained_layout=true)

ax.plot([80, 80], [-100, 100], color="#aaaaaa", linestyle="--")
#ax.plot([180, 180], [-100, 100], color="#aaaaaa", linestyle="--")

for (label, casename) in cases

    println("Case: $casename")
    _d = data[label]

    local rng
    if label == "SIL_SOM"
        rng = 1:10
    elseif label == "SIL_OGCM"
        println("detected $label")
        rng = 1:60
    else
        rng = 1:33
    end

    _TEMP = mean(_d["TEMP"][rng, :], Weights(Δz_cT[rng]), dims=1)[1, :]

    #println(size(_TEMP))

    _TEMP_am = cyclic_mean(_TEMP, 12)

    _TEMP_am .-= _TEMP_am[1]

    #println(size(_TEMP_am))
    Nt = length(_TEMP_am)

    ax.plot(collect(1:Nt).-0.5, _TEMP_am, label=label)

end
    

ax.set_xticks([0, 50, 80, 100, 150, 180, 200, 250, 300])
ax.set_xlim([0, 180])
ax.set_ylim([-0.1, 1.0])
ax.legend()



ax.set_ylabel("\$\\Delta T\$ [\${}^\\circ \\mathrm{C}\$]")
ax.set_xlabel("Year")

fig.savefig("figures/figS02a_ocean_mean_temperature_timeseries.png", dpi=600)

plt.show()
