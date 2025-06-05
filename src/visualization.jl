module Visualization

using Plots
using StatsPlots
using CairoMakie
using DataFrames
using Statistics
using LinearAlgebra

export line_plot, scatter_plot, bar_plot, histogram_plot
export box_plot, violin_plot, heatmap_plot, contour_plot
export surface_plot, density_plot, error_plot, area_plot
export pie_chart, radar_chart, parallel_coordinates
export save_plot, plot_grid, animation

"""
    line_plot(x, y; kwargs...)

Create a line plot.

# Arguments
- `x`: x-coordinates
- `y`: y-coordinates or matrix of y-coordinates
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function line_plot(x, y; kwargs...)
    return plot(x, y; kwargs...)
end

"""
    scatter_plot(x, y; kwargs...)

Create a scatter plot.

# Arguments
- `x`: x-coordinates
- `y`: y-coordinates
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function scatter_plot(x, y; kwargs...)
    return scatter(x, y; kwargs...)
end

"""
    bar_plot(x, y; kwargs...)

Create a bar plot.

# Arguments
- `x`: x-coordinates or categories
- `y`: y-coordinates or heights
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function bar_plot(x, y; kwargs...)
    return bar(x, y; kwargs...)
end

"""
    histogram_plot(x; bins=:auto, kwargs...)

Create a histogram.

# Arguments
- `x`: Data
- `bins`: Number of bins or method to determine bins
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function histogram_plot(x; bins=:auto, kwargs...)
    return histogram(x; bins=bins, kwargs...)
end

"""
    box_plot(x, y; kwargs...)

Create a box plot.

# Arguments
- `x`: Categories
- `y`: Values
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function box_plot(x, y; kwargs...)
    return boxplot(x, y; kwargs...)
end

"""
    violin_plot(x, y; kwargs...)

Create a violin plot.

# Arguments
- `x`: Categories
- `y`: Values
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function violin_plot(x, y; kwargs...)
    return violin(x, y; kwargs...)
end

"""
    heatmap_plot(z; x=nothing, y=nothing, kwargs...)

Create a heatmap.

# Arguments
- `z`: Matrix of values
- `x`: x-coordinates
- `y`: y-coordinates
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function heatmap_plot(z; x=nothing, y=nothing, kwargs...)
    if isnothing(x) && isnothing(y)
        return heatmap(z; kwargs...)
    elseif isnothing(y)
        return heatmap(x, z; kwargs...)
    else
        return heatmap(x, y, z; kwargs...)
    end
end

"""
    contour_plot(z; x=nothing, y=nothing, kwargs...)

Create a contour plot.

# Arguments
- `z`: Matrix of values
- `x`: x-coordinates
- `y`: y-coordinates
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function contour_plot(z; x=nothing, y=nothing, kwargs...)
    if isnothing(x) && isnothing(y)
        return contour(z; kwargs...)
    elseif isnothing(y)
        return contour(x, z; kwargs...)
    else
        return contour(x, y, z; kwargs...)
    end
end

"""
    surface_plot(z; x=nothing, y=nothing, kwargs...)

Create a surface plot.

# Arguments
- `z`: Matrix of values
- `x`: x-coordinates
- `y`: y-coordinates
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function surface_plot(z; x=nothing, y=nothing, kwargs...)
    if isnothing(x) && isnothing(y)
        return surface(z; kwargs...)
    elseif isnothing(y)
        return surface(x, z; kwargs...)
    else
        return surface(x, y, z; kwargs...)
    end
end

"""
    density_plot(x; kwargs...)

Create a density plot.

# Arguments
- `x`: Data
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function density_plot(x; kwargs...)
    return density(x; kwargs...)
end

"""
    error_plot(x, y, yerr; kwargs...)

Create an error plot.

# Arguments
- `x`: x-coordinates
- `y`: y-coordinates
- `yerr`: Error bars
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function error_plot(x, y, yerr; kwargs...)
    return plot(x, y; yerror=yerr, kwargs...)
end

"""
    area_plot(x, y; kwargs...)

Create an area plot.

# Arguments
- `x`: x-coordinates
- `y`: y-coordinates or matrix of y-coordinates
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function area_plot(x, y; kwargs...)
    return plot(x, y; fillrange=0, kwargs...)
end

"""
    pie_chart(labels, values; kwargs...)

Create a pie chart.

# Arguments
- `labels`: Labels for each slice
- `values`: Values for each slice
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function pie_chart(labels, values; kwargs...)
    return pie(labels, values; kwargs...)
end

"""
    radar_chart(labels, values; kwargs...)

Create a radar chart.

# Arguments
- `labels`: Labels for each axis
- `values`: Values for each axis
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function radar_chart(labels, values; kwargs...)
    # Ensure values is a matrix
    if ndims(values) == 1
        values = reshape(values, 1, length(values))
    end
    
    # Number of variables
    n = length(labels)
    
    # Compute angles
    angles = range(0, 2π, length=n+1)[1:end-1]
    
    # Create plot
    p = plot(; aspect_ratio=1, legend=:topright, kwargs...)
    
    # Plot each series
    for i in 1:size(values, 1)
        # Close the polygon
        vals = vcat(values[i, :], values[i, 1])
        angs = vcat(angles, angles[1])
        
        # Convert to Cartesian coordinates
        x = vals .* cos.(angs)
        y = vals .* sin.(angs)
        
        # Plot
        plot!(p, x, y; label="Series $i", kwargs...)
    end
    
    # Add axes
    for (i, angle) in enumerate(angles)
        plot!(p, [0, cos(angle)], [0, sin(angle)]; color=:gray, alpha=0.3, label="", kwargs...)
        annotate!(p, 1.1 * cos(angle), 1.1 * sin(angle), text(labels[i], 8))
    end
    
    return p
end

"""
    parallel_coordinates(df; kwargs...)

Create a parallel coordinates plot.

# Arguments
- `df`: DataFrame
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function parallel_coordinates(df; kwargs...)
    return StatsPlots.parallelplot(df; kwargs...)
end

"""
    save_plot(p, filename; kwargs...)

Save a plot to a file.

# Arguments
- `p`: Plot object
- `filename`: Output filename
- `kwargs...`: Additional arguments to pass to the savefig function
"""
function save_plot(p, filename; kwargs...)
    savefig(p, filename; kwargs...)
end

"""
    plot_grid(plots...; layout=nothing, kwargs...)

Create a grid of plots.

# Arguments
- `plots...`: Plot objects
- `layout`: Layout of the grid (e.g., (2, 2) for a 2x2 grid)
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function plot_grid(plots...; layout=nothing, kwargs...)
    if isnothing(layout)
        n = length(plots)
        ncols = ceil(Int, sqrt(n))
        nrows = ceil(Int, n / ncols)
        layout = (nrows, ncols)
    end
    
    return plot(plots...; layout=layout, kwargs...)
end

"""
    animation(x, y, z=nothing; fps=30, kwargs...)

Create an animation.

# Arguments
- `x`: x-coordinates
- `y`: y-coordinates or matrix of y-coordinates
- `z`: z-coordinates (optional, for 3D animations)
- `fps`: Frames per second
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Animation object
"""
function animation(x, y, z=nothing; fps=30, kwargs...)
    if isnothing(z)
        # 2D animation
        anim = @animate for i in 1:size(y, 2)
            plot(x, y[:, i]; kwargs...)
        end
    else
        # 3D animation
        anim = @animate for i in 1:size(y, 2)
            plot3d(x, y[:, i], z[:, i]; kwargs...)
        end
    end
    
    return gif(anim, fps=fps)
end

end # module

