module DataAnalysis

using Statistics
using StatsBase
using DataFrames
using CSV
using HDF5
using JLD2
using FFTW
using GLM
using Distributions

export load_data, save_data
export descriptive_statistics, correlation_analysis
export hypothesis_testing, anova_test
export linear_regression, multiple_regression
export time_series_analysis, frequency_analysis
export principal_component_analysis, factor_analysis
export bootstrap_analysis, jackknife_analysis

"""
    load_data(filename; format=:auto)

Load data from a file.

# Arguments
- `filename`: Path to the file
- `format`: File format (:csv, :jld2, :hdf5, or :auto to detect from extension)

# Returns
- Loaded data (DataFrame for CSV, dictionary for JLD2/HDF5)
"""
function load_data(filename; format=:auto)
    if format == :auto
        ext = lowercase(splitext(filename)[2])
        if ext == ".csv"
            format = :csv
        elseif ext == ".jld2"
            format = :jld2
        elseif ext == ".h5" || ext == ".hdf5"
            format = :hdf5
        else
            error("Unknown file format: $ext")
        end
    end
    
    if format == :csv
        return CSV.read(filename, DataFrame)
    elseif format == :jld2
        return JLD2.load(filename)
    elseif format == :hdf5
        data = Dict()
        h5open(filename, "r") do file
            for name in names(file)
                data[name] = read(file, name)
            end
        end
        return data
    else
        error("Unsupported format: $format")
    end
end

"""
    save_data(data, filename; format=:auto)

Save data to a file.

# Arguments
- `data`: Data to save (DataFrame for CSV, dictionary for JLD2/HDF5)
- `filename`: Path to the file
- `format`: File format (:csv, :jld2, :hdf5, or :auto to detect from extension)
"""
function save_data(data, filename; format=:auto)
    if format == :auto
        ext = lowercase(splitext(filename)[2])
        if ext == ".csv"
            format = :csv
        elseif ext == ".jld2"
            format = :jld2
        elseif ext == ".h5" || ext == ".hdf5"
            format = :hdf5
        else
            error("Unknown file format: $ext")
        end
    end
    
    if format == :csv
        CSV.write(filename, data)
    elseif format == :jld2
        JLD2.save(filename, data)
    elseif format == :hdf5
        h5open(filename, "w") do file
            for (key, value) in data
                write(file, string(key), value)
            end
        end
    else
        error("Unsupported format: $format")
    end
end

"""
    descriptive_statistics(data)

Compute descriptive statistics for a dataset.

# Arguments
- `data`: Vector, Matrix, or DataFrame

# Returns
- DataFrame with statistics (mean, std, min, max, etc.)
"""
function descriptive_statistics(data::AbstractVector)
    stats = Dict(
        "Mean" => mean(data),
        "Std" => std(data),
        "Min" => minimum(data),
        "Q1" => quantile(data, 0.25),
        "Median" => median(data),
        "Q3" => quantile(data, 0.75),
        "Max" => maximum(data),
        "Skewness" => skewness(data),
        "Kurtosis" => kurtosis(data),
        "N" => length(data),
        "Missing" => count(ismissing, data)
    )
    
    return DataFrame(Statistic = keys(stats), Value = values(stats))
end

function descriptive_statistics(data::AbstractMatrix)
    n_cols = size(data, 2)
    stats = Dict()
    
    for i in 1:n_cols
        col_data = @view data[:, i]
        stats["Column $i"] = Dict(
            "Mean" => mean(col_data),
            "Std" => std(col_data),
            "Min" => minimum(col_data),
            "Q1" => quantile(col_data, 0.25),
            "Median" => median(col_data),
            "Q3" => quantile(col_data, 0.75),
            "Max" => maximum(col_data),
            "Skewness" => skewness(col_data),
            "Kurtosis" => kurtosis(col_data),
            "N" => length(col_data),
            "Missing" => count(ismissing, col_data)
        )
    end
    
    return stats
end

function descriptive_statistics(data::DataFrame)
    stats = Dict()
    
    for col_name in names(data)
        col_data = data[!, col_name]
        if eltype(col_data) <: Number
            stats[col_name] = Dict(
                "Mean" => mean(skipmissing(col_data)),
                "Std" => std(skipmissing(col_data)),
                "Min" => minimum(skipmissing(col_data)),
                "Q1" => quantile(collect(skipmissing(col_data)), 0.25),
                "Median" => median(skipmissing(col_data)),
                "Q3" => quantile(collect(skipmissing(col_data)), 0.75),
                "Max" => maximum(skipmissing(col_data)),
                "Skewness" => skewness(collect(skipmissing(col_data))),
                "Kurtosis" => kurtosis(collect(skipmissing(col_data))),
                "N" => length(col_data),
                "Missing" => count(ismissing, col_data)
            )
        else
            stats[col_name] = Dict(
                "Type" => eltype(col_data),
                "N" => length(col_data),
                "Missing" => count(ismissing, col_data),
                "Unique" => length(unique(skipmissing(col_data)))
            )
        end
    end
    
    return stats
end

"""
    correlation_analysis(data; method=:pearson)

Compute correlation matrix for a dataset.

# Arguments
- `data`: Matrix or DataFrame
- `method`: Correlation method (:pearson, :spearman, or :kendall)

# Returns
- Correlation matrix
"""
function correlation_analysis(data::AbstractMatrix; method=:pearson)
    if method == :pearson
        return cor(data)
    elseif method == :spearman
        return corspearman(data)
    elseif method == :kendall
        return corkendall(data)
    else
        error("Unsupported correlation method: $method")
    end
end

function correlation_analysis(data::DataFrame; method=:pearson)
    # Extract numeric columns
    numeric_cols = names(data)[map(col -> eltype(data[!, col]) <: Number, names(data))]
    
    if isempty(numeric_cols)
        error("No numeric columns found in the DataFrame")
    end
    
    # Convert to matrix
    matrix_data = Matrix(data[!, numeric_cols])
    
    # Compute correlation
    cor_matrix = correlation_analysis(matrix_data, method=method)
    
    # Create DataFrame with column names
    cor_df = DataFrame(cor_matrix, numeric_cols)
    
    # Add column names as a column
    insertcols!(cor_df, 1, :Variable => numeric_cols)
    
    return cor_df
end

"""
    hypothesis_testing(x, y; test=:ttest)

Perform hypothesis testing.

# Arguments
- `x`: First sample
- `y`: Second sample (optional for some tests)
- `test`: Test type (:ttest, :wilcoxon, :kstest, etc.)

# Returns
- Test result (p-value, test statistic, etc.)
"""
function hypothesis_testing(x, y=nothing; test=:ttest)
    if test == :ttest
        if isnothing(y)
            # One-sample t-test against μ=0
            t_stat = mean(x) / (std(x) / sqrt(length(x)))
            df = length(x) - 1
            p_value = 2 * ccdf(TDist(df), abs(t_stat))
            return Dict("test" => "One-sample t-test", "statistic" => t_stat, "p_value" => p_value, "df" => df)
        else
            # Two-sample t-test
            n1, n2 = length(x), length(y)
            m1, m2 = mean(x), mean(y)
            s1, s2 = var(x), var(y)
            
            # Pooled variance
            sp = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            
            # t-statistic
            t_stat = (m1 - m2) / (sp * sqrt(1/n1 + 1/n2))
            
            # Degrees of freedom
            df = n1 + n2 - 2
            
            # p-value
            p_value = 2 * ccdf(TDist(df), abs(t_stat))
            
            return Dict("test" => "Two-sample t-test", "statistic" => t_stat, "p_value" => p_value, "df" => df)
        end
    elseif test == :wilcoxon
        if isnothing(y)
            # Wilcoxon signed-rank test
            # Simplified implementation
            d = x .- 0  # Difference from hypothesized median
            d = d[d .!= 0]  # Remove zeros
            r = rank(abs.(d))
            w_plus = sum(r[d .> 0])
            w_minus = sum(r[d .< 0])
            w = min(w_plus, w_minus)
            n = length(d)
            
            # For large samples, use normal approximation
            if n > 20
                μ = n * (n + 1) / 4
                σ = sqrt(n * (n + 1) * (2n + 1) / 24)
                z = (w - μ) / σ
                p_value = 2 * ccdf(Normal(), abs(z))
            else
                # For small samples, p-value would be computed from tables
                # This is a simplified approximation
                p_value = 2 * min(w_plus, w_minus) / (2^n)
            end
            
            return Dict("test" => "Wilcoxon signed-rank test", "statistic" => w, "p_value" => p_value, "n" => n)
        else
            # Wilcoxon rank-sum test (Mann-Whitney U test)
            # Simplified implementation
            n1, n2 = length(x), length(y)
            ranks = rank(vcat(x, y))
            r1 = sum(ranks[1:n1])
            
            # U statistic
            u1 = r1 - n1 * (n1 + 1) / 2
            u2 = n1 * n2 - u1
            u = min(u1, u2)
            
            # For large samples, use normal approximation
            if min(n1, n2) > 20
                μ = n1 * n2 / 2
                σ = sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                z = (u - μ) / σ
                p_value = 2 * ccdf(Normal(), abs(z))
            else
                # For small samples, p-value would be computed from tables
                # This is a simplified approximation
                p_value = 2 * u / (n1 * n2)
            end
            
            return Dict("test" => "Wilcoxon rank-sum test", "statistic" => u, "p_value" => p_value, "n1" => n1, "n2" => n2)
        end
    elseif test == :kstest
        if isnothing(y)
            # One-sample Kolmogorov-Smirnov test against normal distribution
            # Simplified implementation
            x_std = (x .- mean(x)) ./ std(x)
            n = length(x)
            
            # Empirical CDF
            sorted_x = sort(x_std)
            ecdf = (1:n) ./ n
            
            # Theoretical CDF (standard normal)
            tcdf = cdf.(Normal(), sorted_x)
            
            # KS statistic
            ks_stat = maximum(abs.(ecdf .- tcdf))
            
            # Approximate p-value
            p_value = exp(-2 * n * ks_stat^2)
            
            return Dict("test" => "One-sample Kolmogorov-Smirnov test", "statistic" => ks_stat, "p_value" => p_value, "n" => n)
        else
            # Two-sample Kolmogorov-Smirnov test
            # Simplified implementation
            n1, n2 = length(x), length(y)
            
            # Empirical CDFs
            sorted_x = sort(x)
            sorted_y = sort(y)
            ecdf_x = cumsum(ones(n1)) ./ n1
            ecdf_y = cumsum(ones(n2)) ./ n2
            
            # Merge sorted arrays and track which sample each value came from
            merged = sort(vcat([(v, 1) for v in sorted_x], [(v, 2) for v in sorted_y]))
            
            # Compute CDFs at each point
            cdf1 = zeros(length(merged))
            cdf2 = zeros(length(merged))
            
            idx1 = 0
            idx2 = 0
            for i in 1:length(merged)
                if merged[i][2] == 1
                    idx1 += 1
                    cdf1[i] = idx1 / n1
                    cdf2[i] = idx2 / n2
                else
                    idx2 += 1
                    cdf1[i] = idx1 / n1
                    cdf2[i] = idx2 / n2
                end
            end
            
            # KS statistic
            ks_stat = maximum(abs.(cdf1 .- cdf2))
            
            # Approximate p-value
            n_eff = n1 * n2 / (n1 + n2)
            p_value = exp(-2 * n_eff * ks_stat^2)
            
            return Dict("test" => "Two-sample Kolmogorov-Smirnov test", "statistic" => ks_stat, "p_value" => p_value, "n1" => n1, "n2" => n2)
        end
    else
        error("Unsupported test: $test")
    end
end

"""
    anova_test(groups...)

Perform one-way ANOVA test.

# Arguments
- `groups...`: Two or more groups to compare

# Returns
- ANOVA result (F-statistic, p-value, etc.)
"""
function anova_test(groups...)
    k = length(groups)
    if k < 2
        error("ANOVA requires at least two groups")
    end
    
    # Total number of observations
    n_total = sum(length.(groups))
    
    # Grand mean
    grand_mean = sum(sum.(groups)) / n_total
    
    # Between-group sum of squares
    ss_between = sum(length(g) * (mean(g) - grand_mean)^2 for g in groups)
    
    # Within-group sum of squares
    ss_within = sum(sum((x - mean(g))^2 for x in g) for g in groups)
    
    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k
    
    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # F-statistic
    f_stat = ms_between / ms_within
    
    # p-value
    p_value = ccdf(FDist(df_between, df_within), f_stat)
    
    return Dict(
        "F" => f_stat,
        "p_value" => p_value,
        "df_between" => df_between,
        "df_within" => df_within,
        "ss_between" => ss_between,
        "ss_within" => ss_within,
        "ms_between" => ms_between,
        "ms_within" => ms_within
    )
end

"""
    linear_regression(X, y)

Perform linear regression.

# Arguments
- `X`: Predictor variable(s) (vector or matrix)
- `y`: Response variable

# Returns
- Regression model
"""
function linear_regression(X::AbstractVector, y::AbstractVector)
    df = DataFrame(X = X, y = y)
    model = lm(@formula(y ~ X), df)
    return model
end

function linear_regression(X::AbstractMatrix, y::AbstractVector)
    n_features = size(X, 2)
    df = DataFrame(X, :auto)
    df.y = y
    
    # Create formula
    formula = Term(:y) ~ sum(Term.(Symbol.(names(df)[1:n_features])))
    
    model = lm(formula, df)
    return model
end

"""
    multiple_regression(X, y)

Perform multiple regression.

# Arguments
- `X`: Predictor variables (matrix)
- `y`: Response variable

# Returns
- Regression model
"""
function multiple_regression(X::AbstractMatrix, y::AbstractVector)
    return linear_regression(X, y)
end

"""
    time_series_analysis(data; lag_max=20)

Perform time series analysis.

# Arguments
- `data`: Time series data
- `lag_max`: Maximum lag for autocorrelation

# Returns
- Dictionary with analysis results
"""
function time_series_analysis(data::AbstractVector; lag_max=20)
    n = length(data)
    
    # Compute mean and variance
    μ = mean(data)
    σ² = var(data)
    
    # Compute autocorrelation
    acf = zeros(lag_max + 1)
    for lag in 0:lag_max
        if lag == 0
            acf[lag + 1] = 1.0
        else
            acf[lag + 1] = sum((data[1:n-lag] .- μ) .* (data[lag+1:n] .- μ)) / ((n - lag) * σ²)
        end
    end
    
    # Compute partial autocorrelation
    pacf = zeros(lag_max + 1)
    pacf[1] = 1.0
    
    # Levinson-Durbin recursion for PACF
    for k in 1:lag_max
        if k == 1
            pacf[k + 1] = acf[k + 1]
        else
            # Compute phi_k,k
            phi_kk = (acf[k + 1] - sum(pacf[j + 1] * acf[k - j + 1] for j in 1:k-1)) / 
                     (1 - sum(pacf[j + 1] * acf[j + 1] for j in 1:k-1))
            
            # Update phi_k,j
            phi = zeros(k)
            phi[k] = phi_kk
            for j in 1:k-1
                phi[j] = pacf[j + 1] - phi_kk * pacf[k - j + 1]
            end
            
            # Store phi_k,j
            for j in 1:k
                pacf[j + 1] = phi[j]
            end
        end
    end
    
    # Compute trend
    t = 1:n
    trend_model = lm(@formula(y ~ x), DataFrame(y = data, x = t))
    trend = coef(trend_model)[1] .+ coef(trend_model)[2] .* t
    
    # Compute seasonality (simple moving average)
    window_size = min(n ÷ 4, 12)  # Arbitrary window size
    ma = zeros(n)
    for i in 1:n
        start_idx = max(1, i - window_size ÷ 2)
        end_idx = min(n, i + window_size ÷ 2)
        ma[i] = mean(data[start_idx:end_idx])
    end
    
    seasonality = data .- ma
    
    # Compute residuals
    residuals = data .- trend
    
    return Dict(
        "mean" => μ,
        "variance" => σ²,
        "acf" => acf,
        "pacf" => pacf,
        "trend" => trend,
        "seasonality" => seasonality,
        "residuals" => residuals
    )
end

"""
    frequency_analysis(data; fs=1.0)

Perform frequency analysis using FFT.

# Arguments
- `data`: Time series data
- `fs`: Sampling frequency

# Returns
- Dictionary with frequency analysis results
"""
function frequency_analysis(data::AbstractVector; fs=1.0)
    n = length(data)
    
    # Compute FFT
    fft_result = fft(data)
    
    # Compute power spectrum
    power = abs.(fft_result).^2 / n
    
    # Compute frequencies
    freqs = fs * (0:n-1) / n
    
    # For real data, we only need the first half
    n_half = n ÷ 2 + 1
    power = power[1:n_half]
    freqs = freqs[1:n_half]
    
    # Find dominant frequencies
    sorted_indices = sortperm(power, rev=true)
    top_indices = sorted_indices[1:min(5, length(sorted_indices))]
    dominant_freqs = freqs[top_indices]
    dominant_powers = power[top_indices]
    
    return Dict(
        "frequencies" => freqs,
        "power" => power,
        "dominant_frequencies" => dominant_freqs,
        "dominant_powers" => dominant_powers
    )
end

"""
    principal_component_analysis(X; n_components=nothing)

Perform principal component analysis.

# Arguments
- `X`: Data matrix
- `n_components`: Number of components to keep

# Returns
- Dictionary with PCA results
"""
function principal_component_analysis(X::AbstractMatrix; n_components=nothing)
    # Center the data
    X_centered = X .- mean(X, dims=1)
    
    # Compute covariance matrix
    cov_matrix = (X_centered' * X_centered) ./ (size(X, 1) - 1)
    
    # Compute eigenvalues and eigenvectors
    eigen_vals, eigen_vecs = eigen(cov_matrix)
    
    # Sort by eigenvalues in descending order
    idx = sortperm(eigen_vals, rev=true)
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:, idx]
    
    # Determine number of components
    if isnothing(n_components)
        # Keep components that explain 95% of variance
        explained_var = cumsum(eigen_vals) / sum(eigen_vals)
        n_components = findfirst(explained_var .>= 0.95)
    end
    
    # Select top components
    components = eigen_vecs[:, 1:n_components]
    
    # Project data onto components
    projected = X_centered * components
    
    # Compute explained variance
    explained_var = eigen_vals / sum(eigen_vals)
    
    return Dict(
        "components" => components,
        "explained_variance" => eigen_vals,
        "explained_variance_ratio" => explained_var,
        "singular_values" => sqrt.(eigen_vals * (size(X, 1) - 1)),
        "projected_data" => projected
    )
end

"""
    factor_analysis(X; n_factors=2, max_iter=100, tol=1e-4)

Perform factor analysis.

# Arguments
- `X`: Data matrix
- `n_factors`: Number of factors
- `max_iter`: Maximum number of iterations
- `tol`: Tolerance for convergence

# Returns
- Dictionary with factor analysis results
"""
function factor_analysis(X::AbstractMatrix; n_factors=2, max_iter=100, tol=1e-4)
    n_samples, n_features = size(X)
    
    # Standardize the data
    X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    
    # Initial communalities (using squared multiple correlations)
    R = cor(X_std)
    h2 = zeros(n_features)
    for i in 1:n_features
        R_i = R[setdiff(1:n_features, i), setdiff(1:n_features, i)]
        r_i = R[i, setdiff(1:n_features, i)]
        h2[i] = r_i' * inv(R_i) * r_i
    end
    
    # Initialize uniquenesses
    psi = 1.0 .- h2
    
    # Iterative estimation
    old_psi = copy(psi)
    for iter in 1:max_iter
        # Compute factor loadings
        R_psi = R - Diagonal(psi)
        eigen_vals, eigen_vecs = eigen(R_psi)
        
        # Sort by eigenvalues in descending order
        idx = sortperm(eigen_vals, rev=true)
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]
        
        # Select top factors
        loadings = eigen_vecs[:, 1:n_factors] .* sqrt.(eigen_vals[1:n_factors]')
        
        # Update communalities and uniquenesses
        h2 = sum(loadings.^2, dims=2)
        psi = 1.0 .- h2
        
        # Check convergence
        if maximum(abs.(psi - old_psi)) < tol
            break
        end
        
        old_psi = copy(psi)
    end
    
    # Compute factor scores
    factor_scores = X_std * loadings * inv(loadings' * loadings)
    
    return Dict(
        "loadings" => loadings,
        "uniquenesses" => psi,
        "communalities" => h2,
        "factor_scores" => factor_scores
    )
end

"""
    bootstrap_analysis(data, statistic; n_resamples=1000, α=0.05)

Perform bootstrap analysis.

# Arguments
- `data`: Data vector
- `statistic`: Function to compute the statistic
- `n_resamples`: Number of bootstrap resamples
- `α`: Significance level for confidence intervals

# Returns
- Dictionary with bootstrap results
"""
function bootstrap_analysis(data::AbstractVector, statistic; n_resamples=1000, α=0.05)
    n = length(data)
    bootstrap_stats = zeros(n_resamples)
    
    for i in 1:n_resamples
        # Resample with replacement
        resample = data[rand(1:n, n)]
        
        # Compute statistic
        bootstrap_stats[i] = statistic(resample)
    end
    
    # Compute bootstrap estimate and standard error
    bootstrap_estimate = mean(bootstrap_stats)
    bootstrap_se = std(bootstrap_stats)
    
    # Compute confidence intervals
    # Percentile method
    lower_percentile = α / 2 * 100
    upper_percentile = (1 - α / 2) * 100
    ci_percentile = quantile(bootstrap_stats, [lower_percentile/100, upper_percentile/100])
    
    # BCa method (simplified)
    # Bias correction
    z0 = quantile(Normal(), sum(bootstrap_stats .< statistic(data)) / n_resamples)
    
    # Acceleration (jackknife influence values)
    jackknife_stats = zeros(n)
    for i in 1:n
        jackknife_sample = data[setdiff(1:n, i)]
        jackknife_stats[i] = statistic(jackknife_sample)
    end
    
    jackknife_mean = mean(jackknife_stats)
    a = sum((jackknife_mean .- jackknife_stats).^3) / (6 * sum((jackknife_mean .- jackknife_stats).^2)^(3/2))
    
    # BCa intervals
    z_α1 = quantile(Normal(), α / 2)
    z_α2 = quantile(Normal(), 1 - α / 2)
    
    p1 = cdf(Normal(), z0 + (z0 + z_α1) / (1 - a * (z0 + z_α1)))
    p2 = cdf(Normal(), z0 + (z0 + z_α2) / (1 - a * (z0 + z_α2)))
    
    ci_bca = quantile(bootstrap_stats, [p1, p2])
    
    return Dict(
        "bootstrap_estimate" => bootstrap_estimate,
        "bootstrap_se" => bootstrap_se,
        "ci_percentile" => ci_percentile,
        "ci_bca" => ci_bca,
        "bootstrap_distribution" => bootstrap_stats
    )
end

"""
    jackknife_analysis(data, statistic)

Perform jackknife analysis.

# Arguments
- `data`: Data vector
- `statistic`: Function to compute the statistic

# Returns
- Dictionary with jackknife results
"""
function jackknife_analysis(data::AbstractVector, statistic)
    n = length(data)
    jackknife_stats = zeros(n)
    
    for i in 1:n
        # Leave-one-out sample
        jackknife_sample = data[setdiff(1:n, i)]
        
        # Compute statistic
        jackknife_stats[i] = statistic(jackknife_sample)
    end
    
    # Original statistic
    original_stat = statistic(data)
    
    # Jackknife estimate
    jackknife_estimate = n * original_stat - (n - 1) * mean(jackknife_stats)
    
    # Jackknife standard error
    jackknife_se = sqrt((n - 1) * mean((jackknife_stats .- mean(jackknife_stats)).^2))
    
    # Bias estimate
    bias = (n - 1) * (mean(jackknife_stats) - original_stat)
    
    return Dict(
        "jackknife_estimate" => jackknife_estimate,
        "jackknife_se" => jackknife_se,
        "bias" => bias,
        "jackknife_values" => jackknife_stats
    )
end

end # module

