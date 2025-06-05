module Utilities

using Statistics
using LinearAlgebra
using Random
using Distributions
using DataFrames
using CSV
using HDF5
using JLD2
using BenchmarkTools

export generate_data, load_data, save_data
export benchmark_function, profile_function
export parallel_map, parallel_reduce
export moving_average, exponential_moving_average
export resample_data, bootstrap_sample
export cross_validation_split, stratified_split
export normalize_data, standardize_data
export one_hot_encode, label_encode

"""
    generate_data(n, p=1; dist=:normal, params=nothing, seed=nothing)

Generate random data.

# Arguments
- `n`: Number of samples
- `p`: Number of features
- `dist`: Distribution (:normal, :uniform, :exponential, etc.)
- `params`: Distribution parameters
- `seed`: Random seed

# Returns
- Generated data
"""
function generate_data(n, p=1; dist=:normal, params=nothing, seed=nothing)
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Generate data
    if dist == :normal
        if isnothing(params)
            params = (0.0, 1.0)  # Default: mean=0, std=1
        end
        μ, σ = params
        data = randn(n, p) .* σ .+ μ
    elseif dist == :uniform
        if isnothing(params)
            params = (0.0, 1.0)  # Default: min=0, max=1
        end
        a, b = params
        data = rand(n, p) .* (b - a) .+ a
    elseif dist == :exponential
        if isnothing(params)
            params = 1.0  # Default: rate=1
        end
        λ = params
        data = randexp(n, p) ./ λ
    elseif dist == :poisson
        if isnothing(params)
            params = 1.0  # Default: lambda=1
        end
        λ = params
        data = rand(Poisson(λ), n, p)
    elseif dist == :bernoulli
        if isnothing(params)
            params = 0.5  # Default: p=0.5
        end
        p_success = params
        data = rand(Bernoulli(p_success), n, p)
    elseif dist == :categorical
        if isnothing(params)
            params = [0.25, 0.25, 0.25, 0.25]  # Default: equal probabilities
        end
        probs = params
        data = rand(Categorical(probs), n, p)
    elseif dist == :multivariate_normal
        if isnothing(params)
            params = (zeros(p), Matrix{Float64}(I, p, p))  # Default: mean=0, cov=I
        end
        μ, Σ = params
        data = rand(MvNormal(μ, Σ), n)'
    else
        error("Unknown distribution: $dist")
    end
    
    return data
end

"""
    load_data(filename; format=:auto)

Load data from a file.

# Arguments
- `filename`: Path to the file
- `format`: File format (:csv, :jld2, :hdf5, or :auto to detect from extension)

# Returns
- Loaded data
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
- `data`: Data to save
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
        if data isa DataFrame
            CSV.write(filename, data)
        else
            CSV.write(filename, DataFrame(data))
        end
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
    benchmark_function(f, args...; kwargs...)

Benchmark a function.

# Arguments
- `f`: Function to benchmark
- `args...`: Arguments to pass to the function
- `kwargs...`: Keyword arguments to pass to the function

# Returns
- Benchmark results
"""
function benchmark_function(f, args...; kwargs...)
    return @benchmark $f($(args)...; $(kwargs)...)
end

"""
    profile_function(f, args...; kwargs...)

Profile a function.

# Arguments
- `f`: Function to profile
- `args...`: Arguments to pass to the function
- `kwargs...`: Keyword arguments to pass to the function
"""
function profile_function(f, args...; kwargs...)
    # Clear profile buffer
    Profile.clear()
    
    # Start profiling
    Profile.@profile f(args...; kwargs...)
    
    # Print profile
    Profile.print()
end

"""
    parallel_map(f, collection; batch_size=nothing)

Apply a function to each element of a collection in parallel.

# Arguments
- `f`: Function to apply
- `collection`: Collection of elements
- `batch_size`: Batch size for parallel processing

# Returns
- Result of applying the function to each element
"""
function parallel_map(f, collection; batch_size=nothing)
    # Determine batch size
    if isnothing(batch_size)
        batch_size = max(1, length(collection) ÷ Threads.nthreads())
    end
    
    # Create batches
    n = length(collection)
    n_batches = ceil(Int, n / batch_size)
    batches = [collection[(i-1)*batch_size+1:min(i*batch_size, n)] for i in 1:n_batches]
    
    # Process batches in parallel
    results = Vector{Any}(undef, n_batches)
    Threads.@threads for i in 1:n_batches
        results[i] = map(f, batches[i])
    end
    
    # Combine results
    return vcat(results...)
end

"""
    parallel_reduce(f, op, collection; batch_size=nothing, init=nothing)

Apply a function to each element of a collection and reduce the results in parallel.

# Arguments
- `f`: Function to apply
- `op`: Binary operator for reduction
- `collection`: Collection of elements
- `batch_size`: Batch size for parallel processing
- `init`: Initial value for reduction

# Returns
- Result of reduction
"""
function parallel_reduce(f, op, collection; batch_size=nothing, init=nothing)
    # Determine batch size
    if isnothing(batch_size)
        batch_size = max(1, length(collection) ÷ Threads.nthreads())
    end
    
    # Create batches
    n = length(collection)
    n_batches = ceil(Int, n / batch_size)
    batches = [collection[(i-1)*batch_size+1:min(i*batch_size, n)] for i in 1:n_batches]
    
    # Process batches in parallel
    results = Vector{Any}(undef, n_batches)
    Threads.@threads for i in 1:n_batches
        if isnothing(init)
            results[i] = reduce(op, map(f, batches[i]))
        else
            results[i] = reduce(op, map(f, batches[i]), init=init)
        end
    end
    
    # Combine results
    if isnothing(init)
        return reduce(op, results)
    else
        return reduce(op, results, init=init)
    end
end

"""
    moving_average(data, window_size)

Compute moving average.

# Arguments
- `data`: Input data
- `window_size`: Window size

# Returns
- Moving average
"""
function moving_average(data, window_size)
    n = length(data)
    result = similar(data)
    
    for i in 1:n
        start_idx = max(1, i - window_size ÷ 2)
        end_idx = min(n, i + window_size ÷ 2)
        result[i] = mean(data[start_idx:end_idx])
    end
    
    return result
end

"""
    exponential_moving_average(data, α)

Compute exponential moving average.

# Arguments
- `data`: Input data
- `α`: Smoothing factor (0 < α < 1)

# Returns
- Exponential moving average
"""
function exponential_moving_average(data, α)
    n = length(data)
    result = similar(data)
    
    result[1] = data[1]
    for i in 2:n
        result[i] = α * data[i] + (1 - α) * result[i-1]
    end
    
    return result
end

"""
    resample_data(data, n=nothing; replace=true, weights=nothing, seed=nothing)

Resample data.

# Arguments
- `data`: Input data
- `n`: Number of samples (default: length(data))
- `replace`: Whether to sample with replacement
- `weights`: Sampling weights
- `seed`: Random seed

# Returns
- Resampled data
"""
function resample_data(data, n=nothing; replace=true, weights=nothing, seed=nothing)
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Determine number of samples
    if isnothing(n)
        n = length(data)
    end
    
    # Sample indices
    indices = sample(1:length(data), weights, n, replace=replace)
    
    # Return resampled data
    return data[indices]
end

"""
    bootstrap_sample(data, n_samples=1000; seed=nothing)

Generate bootstrap samples.

# Arguments
- `data`: Input data
- `n_samples`: Number of bootstrap samples
- `seed`: Random seed

# Returns
- Bootstrap samples
"""
function bootstrap_sample(data, n_samples=1000; seed=nothing)
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Generate bootstrap samples
    n = length(data)
    samples = [resample_data(data, n, replace=true) for _ in 1:n_samples]
    
    return samples
end

"""
    cross_validation_split(data, n_folds=5; shuffle=true, seed=nothing)

Split data for cross-validation.

# Arguments
- `data`: Input data
- `n_folds`: Number of folds
- `shuffle`: Whether to shuffle the data
- `seed`: Random seed

# Returns
- List of (train_indices, test_indices) pairs
"""
function cross_validation_split(data, n_folds=5; shuffle=true, seed=nothing)
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Determine indices
    n = length(data)
    indices = 1:n
    
    if shuffle
        indices = shuffle(indices)
    end
    
    # Create folds
    fold_size = n ÷ n_folds
    splits = []
    
    for i in 1:n_folds
        start_idx = (i - 1) * fold_size + 1
        end_idx = i == n_folds ? n : i * fold_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = setdiff(indices, test_indices)
        
        push!(splits, (train_indices, test_indices))
    end
    
    return splits
end

"""
    stratified_split(data, labels, n_folds=5; shuffle=true, seed=nothing)

Split data for stratified cross-validation.

# Arguments
- `data`: Input data
- `labels`: Labels for stratification
- `n_folds`: Number of folds
- `shuffle`: Whether to shuffle the data
- `seed`: Random seed

# Returns
- List of (train_indices, test_indices) pairs
"""
function stratified_split(data, labels, n_folds=5; shuffle=true, seed=nothing)
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Group indices by label
    label_indices = Dict()
    for (i, label) in enumerate(labels)
        if !haskey(label_indices, label)
            label_indices[label] = []
        end
        push!(label_indices[label], i)
    end
    
    # Shuffle indices within each label
    if shuffle
        for label in keys(label_indices)
            label_indices[label] = shuffle(label_indices[label])
        end
    end
    
    # Create folds
    splits = []
    
    for i in 1:n_folds
        test_indices = []
        
        # Add indices from each label
        for (label, indices) in label_indices
            n_label = length(indices)
            fold_size = n_label ÷ n_folds
            
            start_idx = (i - 1) * fold_size + 1
            end_idx = i == n_folds ? n_label : i * fold_size
            
            append!(test_indices, indices[start_idx:end_idx])
        end
        
        train_indices = setdiff(1:length(data), test_indices)
        
        push!(splits, (train_indices, test_indices))
    end
    
    return splits
end

"""
    normalize_data(X; dims=1)

Normalize data to [0, 1] range.

# Arguments
- `X`: Data matrix
- `dims`: Dimension along which to normalize (1 for columns, 2 for rows)

# Returns
- Normalized data and normalization parameters
"""
function normalize_data(X; dims=1)
    X_min = minimum(X, dims=dims)
    X_max = maximum(X, dims=dims)
    
    # Avoid division by zero
    X_range = X_max - X_min
    X_range[X_range .== 0] .= 1.0
    
    X_norm = (X .- X_min) ./ X_range
    
    return X_norm, (X_min, X_range)
end

"""
    standardize_data(X; dims=1)

Standardize data to zero mean and unit variance.

# Arguments
- `X`: Data matrix
- `dims`: Dimension along which to standardize (1 for columns, 2 for rows)

# Returns
- Standardized data and standardization parameters
"""
function standardize_data(X; dims=1)
    X_mean = mean(X, dims=dims)
    X_std = std(X, dims=dims)
    
    # Avoid division by zero
    X_std[X_std .== 0] .= 1.0
    
    X_stand = (X .- X_mean) ./ X_std
    
    return X_stand, (X_mean, X_std)
end

"""
    one_hot_encode(labels)

One-hot encode categorical labels.

# Arguments
- `labels`: Categorical labels

# Returns
- One-hot encoded labels and mapping
"""
function one_hot_encode(labels)
    # Get unique labels
    unique_labels = sort(unique(labels))
    n_labels = length(unique_labels)
    
    # Create mapping
    label_to_index = Dict(label => i for (i, label) in enumerate(unique_labels))
    
    # Create one-hot encoded matrix
    n = length(labels)
    encoded = zeros(n, n_labels)
    
    for (i, label) in enumerate(labels)
        encoded[i, label_to_index[label]] = 1.0
    end
    
    return encoded, label_to_index
end

"""
    label_encode(labels)

Encode categorical labels as integers.

# Arguments
- `labels`: Categorical labels

# Returns
- Integer-encoded labels and mapping
"""
function label_encode(labels)
    # Get unique labels
    unique_labels = sort(unique(labels))
    
    # Create mapping
    label_to_index = Dict(label => i for (i, label) in enumerate(unique_labels))
    
    # Encode labels
    encoded = [label_to_index[label] for label in labels]
    
    return encoded, label_to_index
end

end # module

