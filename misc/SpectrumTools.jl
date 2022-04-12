module SpectrumTools

    export hfft, hifft, genBandpassFilter, mavg

    using FFTW

    function hfft(y)
        spec = fft(y)
        return spec[1:Int64(length(spec)/2)+1]
    end

    function hifft(hspec)
        spec = zeros(Complex{Float64}, (length(hspec)-1)*2)
        spec[1:length(hspec)] = hspec
        spec[length(hspec)+1:end] = reverse(conj.(hspec[2:length(hspec)-1]))
        return real.(ifft(spec))
    end


    function genBandpassFilter(f, cutoff_lower, cutoff_upper, σ_f)

        local bpf_upper = exp.( -  ((f.-cutoff_lower) / σ_f).^2.0)
        local bpf_lower = exp.( -  ((f.-cutoff_upper) / σ_f).^2.0)
        bpf_upper[f .> cutoff_lower] .= 1.0
        bpf_lower[f .< cutoff_upper] .= 1.0

        return bpf_upper .* bpf_lower
    end



    function mavg(x, n)

        _x = zeros(Float64, length(x))

        for i = 1:length(_x)
            l_idx = max(i - n, 1)
            r_idx = min(i + n, length(_x))

            _x[i] = sum(x[l_idx:r_idx]) / (r_idx - l_idx + 1)

        end

        return _x

    end
end
