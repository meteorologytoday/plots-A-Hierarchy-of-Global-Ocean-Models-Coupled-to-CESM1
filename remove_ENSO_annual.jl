using NCDatasets
using PyCall
using Statistics

function rmSignal(data, signal)

    if length(data) != length(signal)
        print("Length of data=$(length(data)) length of signal=$(length(signal))")
        throw(ErrorException("Unequal Length."))
    end

    data_a = data .- mean(data)
    signal_a = signal .- mean(signal)

    return data_a - ( sum(data_a .* signal_a) / sum(signal_a .* signal_a)) * signal_a
               
end

ENSO_file = "data/CTL_21-120/POP2_CTL/atm_analysis_ENSO.nc"


modes_removed = 2

println("Getting cases...")
pushfirst!(PyVector(pyimport("sys")."path"), ".")
tools = pyimport("quick_tools")
sim_cases = tools.getSimcases(["OGCM"])
sim_vars = tools.getSimVars(["SST", "PSL", "TREFHT"])
println("Done")

avg_time = ["season", "annual"][1]


if avg_time == "season"
    roll_offset = -2
    avg_len = 3
elseif avg_time == "annual"
    roll_offset = 0
    avg_len = 12
end

annual_variability_jump = Int64(12 / avg_len)


println("Removing signal of ENSO out of OGCM")


println("1. Load data")
Dataset(ENSO_file, "r") do ds
    global _ENSO_idx = nomissing(ds["PCAs_ts"][1:modes_removed, :], NaN)
end


for varname in keys(sim_vars)


    output_file = "OGCM_$(varname)A_Statistics_remove_ENSO.nc"
    
    println("Doing varname: $varname with output file $output_file")    

    Dataset(joinpath("data", sim_cases["OGCM"]["CTL"], sim_vars[varname]), "r") do ds
        global OGCM_VAR_MA = nomissing(ds["$(varname)_MA"][:, :, 1, :], NaN)
    end

    #println(size(OGCM_VAR_MA))
    Nx, Ny, Nt = size(OGCM_VAR_MA)

    VARA_std          = zeros(Float64, Nx, Ny, annual_variability_jump)
    VARA_noENSO_std   = VARA_std * 0.0

    ENSO_idx = zeros(Float64, modes_removed, Int64(Nt/avg_len)) 
    for m=1:modes_removed
        ENSO_idx[m, :] = mean(reshape(circshift(view(_ENSO_idx, m, :), (roll_offset,)), avg_len, :), dims=1)[1, :]
    end
    #ENSO_idx = mean(reshape(circshift(ENSO_idx, (roll_offset,)), avg_len, :), dims=1)[1, :]

    println("2. Remove signal")
    @time for i=1:Nx, j=1:Ny
        VARA = mean(reshape(circshift(view(OGCM_VAR_MA, i, j, :), (roll_offset,)), avg_len, :), dims=1)[1, :]

        local VARA_rm
            
        VARA_rm = VARA * 1.0
        for m = 1:modes_removed
            VARA_rm = rmSignal(VARA_rm, view(ENSO_idx, m, :))
        end
       
        VARA_std[i, j, :]        = std(reshape(VARA,    annual_variability_jump, :), dims=2)[:, 1]
        VARA_noENSO_std[i, j, :] = std(reshape(VARA_rm, annual_variability_jump, :), dims=2)[:, 1]
    end
    println("Done")

    Dataset(output_file, "c") do ds


        defDim(ds, "Nx", Nx) 
        defDim(ds, "Ny", Ny) 
        defDim(ds, "time", Inf) 

        for (varname, vardata, vardim, attrib) in [
            ("VARA_STD", VARA_std, ("Nx", "Ny", "time",), Dict()),
            ("VARA_noENSO_STD", VARA_noENSO_std, ("Nx", "Ny", "time",), Dict()),
        ]   

            if ! haskey(ds, varname)
                var = defVar(ds, varname, Float64, vardim)
                var.attrib["_FillValue"] = 1e20
            end 

            var = ds[varname]
                
            for (k, v) in attrib
                var.attrib[k] = v 
            end 

            rng = []
            for i in 1:length(vardim)-1
                push!(rng, Colon())
            end 
            push!(rng, 1:size(vardata)[end])
            var[rng...] = vardata

        end 

    end

end
