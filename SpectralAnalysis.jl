module SpectralAnalysis

    using Distributions: Chisq, quantile, cquantile

    export computeSpectrum, computeCIRatio



    function correlation(
        arr :: AbstractArray{T},
        lag :: Integer,
    ) where T <: AbstractFloat

        N = length(arr)

        if lag >= N
            throw(ErrorException("Lag should be shorter than the length of the array."))
        end

        return transpose(view(arr, 1:(N-lag))) * view(arr, (lag+1):N) / N

    end


    function computeSpectrum(
        arr :: AbstractArray{T};
        smoothing :: String = "none",
        lag_window_M :: Int64 = -1,
    ) where T <: AbstractFloat

        N = length(arr)

        if mod(N, 2) != 0
            throw(ErrorException("Array length should be an even number."))
        end

        sin_vecs, cos_vecs = spectralVecs(N; dtype=T)
        
        c0 = correlation(arr, 0)
        c = zeros(T, N-1)
        for i=1:N-1
            c[i] = correlation(arr, i)
        end
 
        spec = zeros(T, Int64(N/2))

        if smoothing == "none"
            λ = ones(T, N)
        elseif smoothing in ["Tukey", "Parzen"]
            lag_window_M = (lag_window_M == -1) ? floor(Int64, N/3) : lag_window_M
            λ = _lag_window(N, lag_window_M, smoothing; dtype=T)
        else
            throw(ErrorException("Unknown smoothing method: $(smoothing). Only 'none', 'Tukey', and 'Parzen' are allowed."))
        end

        λ0 = λ[1]
        λ_vec = view(λ, 2:length(λ))

        println("Length of c = $(length(c))")
        for p=1:length(spec)
            spec[p] = ( c0 * λ0 + 2 * sum(view(cos_vecs, 1:N-1, p) .* c .* λ_vec) ) / π
        end

        

        return spec, _dω(N; dtype=T), λ 
    end

    function _dω(
        N :: Int64;
        dtype :: Type = Float64,
    )

        dω = zeros(dtype, Int64(N/2))
        dω[1:end-1] .= 2π / N
        dω[end] = π / N

        return dω
    end

    function _lag_window(
        N :: Int64,
        M :: Int64,
        smoothing :: String;
        dtype :: Type = Float64,
    )
        if M >= N || M < 1
            throw(ErrorException("M should satisfy 0 < M < N."))
        end

       
        λ = zeros(dtype, N)

        if smoothing == "Tukey"
            k = collect(dtype, 0:M)
            λ[1:M+1] .= (1 .+ cos.(π * k / M)) / 2
        elseif smoothing == "Parzen"

            if mod(M, 2) != 0
                if M + 1 < N
                    M += 1
                else
                    M -= 1
                end
            end 
     
            for i=1:length(λ)
                k = i-1
                if k <= M/2
                    λ[i] = 1.0 - 6 * (k/M)^2 + 6 * (k/M)^3
                else
                    λ[i] = 2 * (1.0 - k/M)^3
                end
            end

        else
            throw(ErrorException("Unknown smoothing method: $smoothing."))
        end


        return λ
    end

    function spectralVecs(
        N :: Integer;
        dtype :: Type = Float64,
    )
        if dtype <: Real
            vecs = zeros(dtype, N, N)
        end
            
        p_vec = reshape(collect(dtype, 1:(N/2)), 1, :)
        t_vec = reshape(collect(dtype, 1:N),     :, 1)

        mtx = (2π/N) * p_vec .* t_vec

        cos_vecs = cos.(mtx)
        sin_vecs = sin.(mtx)
        
        return sin_vecs, cos_vecs
    end

    function computeCIRatio(
        N :: Integer,
        λ :: AbstractArray{T};
        α :: Float64 = 0.05,
    ) where T <: Real
        
        if ! ( 0 < α < 1)
            throw(ErrorException("α value should be between 0 and 1."))
        end

        # degree of freedom
        ν = 2 * N / (sum(λ[2:end].^2) + λ[1]^2)

        χ2dist = Chisq(ν)
        χ2_lower = cquantile(χ2dist,       α/2) 
        χ2_upper = cquantile(χ2dist, 1.0 - α/2) 
        
        return ν/χ2_lower, ν/χ2_upper
    end

end
