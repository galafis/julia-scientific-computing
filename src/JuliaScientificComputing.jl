module JuliaScientificComputing

using LinearAlgebra
using Statistics
using Random
using Distributions
using DataFrames
using CSV
using HDF5
using JLD2
using DifferentialEquations
using Optim
using FFTW
using StatsBase
using GLM
using MLJ
using Plots
using StatsPlots
using CairoMakie
using BenchmarkTools

# Export modules
export NumericalMethods
export DataAnalysis
export DynamicalSystems
export MachineLearning
export Visualization
export Utilities

# Include submodules
include("numerical_methods.jl")
include("data_analysis.jl")
include("dynamical_systems.jl")
include("machine_learning.jl")
include("visualization.jl")
include("utilities.jl")

end # module

