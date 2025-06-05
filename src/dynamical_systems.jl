module DynamicalSystems

using DifferentialEquations
using LinearAlgebra
using Statistics
using Random
using Distributions
using Plots

export solve_ode, solve_sde, solve_dde
export lorenz_system, rossler_system, double_pendulum
export phase_space_plot, poincare_section
export lyapunov_exponent, bifurcation_diagram
export basin_of_attraction, fractal_dimension

"""
    solve_ode(f, u0, tspan; kwargs...)

Solve an ordinary differential equation.

# Arguments
- `f`: Function defining the ODE (du/dt = f(u, p, t))
- `u0`: Initial condition
- `tspan`: Time span (t0, tf)
- `kwargs...`: Additional arguments to pass to the solver

# Returns
- Solution object
"""
function solve_ode(f, u0, tspan; kwargs...)
    prob = ODEProblem(f, u0, tspan)
    return solve(prob; kwargs...)
end

"""
    solve_sde(f, g, u0, tspan; kwargs...)

Solve a stochastic differential equation.

# Arguments
- `f`: Drift function (du = f(u, p, t)dt)
- `g`: Diffusion function (du = g(u, p, t)dW)
- `u0`: Initial condition
- `tspan`: Time span (t0, tf)
- `kwargs...`: Additional arguments to pass to the solver

# Returns
- Solution object
"""
function solve_sde(f, g, u0, tspan; kwargs...)
    prob = SDEProblem(f, g, u0, tspan)
    return solve(prob; kwargs...)
end

"""
    solve_dde(f, u0, h, tspan; kwargs...)

Solve a delay differential equation.

# Arguments
- `f`: Function defining the DDE (du/dt = f(u, h, p, t))
- `u0`: Initial condition
- `h`: History function
- `tspan`: Time span (t0, tf)
- `kwargs...`: Additional arguments to pass to the solver

# Returns
- Solution object
"""
function solve_dde(f, u0, h, tspan; kwargs...)
    prob = DDEProblem(f, u0, h, tspan)
    return solve(prob; kwargs...)
end

"""
    lorenz_system(u, p, t)

Lorenz system of differential equations.

# Arguments
- `u`: State vector [x, y, z]
- `p`: Parameters [σ, ρ, β]
- `t`: Time

# Returns
- Derivative vector [dx/dt, dy/dt, dz/dt]
"""
function lorenz_system(u, p, t)
    σ, ρ, β = p
    x, y, z = u
    
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
    
    return [dx, dy, dz]
end

"""
    rossler_system(u, p, t)

Rössler system of differential equations.

# Arguments
- `u`: State vector [x, y, z]
- `p`: Parameters [a, b, c]
- `t`: Time

# Returns
- Derivative vector [dx/dt, dy/dt, dz/dt]
"""
function rossler_system(u, p, t)
    a, b, c = p
    x, y, z = u
    
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    
    return [dx, dy, dz]
end

"""
    double_pendulum(u, p, t)

Double pendulum system of differential equations.

# Arguments
- `u`: State vector [θ₁, θ₂, ω₁, ω₂]
- `p`: Parameters [m₁, m₂, l₁, l₂, g]
- `t`: Time

# Returns
- Derivative vector [dθ₁/dt, dθ₂/dt, dω₁/dt, dω₂/dt]
"""
function double_pendulum(u, p, t)
    m1, m2, l1, l2, g = p
    θ1, θ2, ω1, ω2 = u
    
    # Compute derivatives
    dθ1 = ω1
    dθ2 = ω2
    
    # Compute auxiliary terms
    Δθ = θ2 - θ1
    den = (m1 + m2) * l1 - m2 * l1 * cos(Δθ)^2
    
    # Compute angular accelerations
    dω1 = (m2 * l1 * ω1^2 * sin(Δθ) * cos(Δθ) +
           m2 * g * sin(θ2) * cos(Δθ) +
           m2 * l2 * ω2^2 * sin(Δθ) -
           (m1 + m2) * g * sin(θ1)) / den
    
    dω2 = (-m2 * l2 * ω2^2 * sin(Δθ) * cos(Δθ) +
           (m1 + m2) * g * sin(θ1) * cos(Δθ) -
           (m1 + m2) * l1 * ω1^2 * sin(Δθ) -
           (m1 + m2) * g * sin(θ2)) / (m2 * l2 * den / l1)
    
    return [dθ1, dθ2, dω1, dω2]
end

"""
    phase_space_plot(sol, dims=[1, 2, 3]; kwargs...)

Create a phase space plot from a solution.

# Arguments
- `sol`: Solution object
- `dims`: Dimensions to plot
- `kwargs...`: Additional arguments to pass to the plot function

# Returns
- Plot object
"""
function phase_space_plot(sol, dims=[1, 2, 3]; kwargs...)
    if length(dims) == 2
        return plot(sol[dims[1], :], sol[dims[2], :]; xlabel="x$(dims[1])", ylabel="x$(dims[2])", kwargs...)
    elseif length(dims) == 3
        return plot3d(sol[dims[1], :], sol[dims[2], :], sol[dims[3], :]; xlabel="x$(dims[1])", ylabel="x$(dims[2])", zlabel="x$(dims[3])", kwargs...)
    else
        error("dims must have length 2 or 3")
    end
end

"""
    poincare_section(sol, plane_dim, plane_value, dims=[1, 2])

Compute a Poincaré section of a solution.

# Arguments
- `sol`: Solution object
- `plane_dim`: Dimension of the plane
- `plane_value`: Value of the plane
- `dims`: Dimensions to plot

# Returns
- Poincaré section points
"""
function poincare_section(sol, plane_dim, plane_value, dims=[1, 2])
    # Extract solution data
    t = sol.t
    u = sol.u
    
    # Initialize arrays for Poincaré section
    poincare_points = []
    
    # Find crossings of the plane
    for i in 2:length(t)
        u1 = u[i-1][plane_dim]
        u2 = u[i][plane_dim]
        
        # Check if the trajectory crosses the plane
        if (u1 - plane_value) * (u2 - plane_value) <= 0 && (u2 - u1) > 0
            # Linear interpolation to find the crossing point
            α = (plane_value - u1) / (u2 - u1)
            crossing_point = u[i-1] + α * (u[i] - u[i-1])
            
            # Extract the dimensions of interest
            point = [crossing_point[d] for d in dims]
            push!(poincare_points, point)
        end
    end
    
    return poincare_points
end

"""
    lyapunov_exponent(f, u0, p, tspan, n_steps=1000, n_vectors=10, n_iterations=100)

Compute the Lyapunov exponents of a dynamical system.

# Arguments
- `f`: Function defining the ODE (du/dt = f(u, p, t))
- `u0`: Initial condition
- `p`: Parameters
- `tspan`: Time span (t0, tf)
- `n_steps`: Number of time steps
- `n_vectors`: Number of orthogonal vectors to use
- `n_iterations`: Number of iterations for convergence

# Returns
- Lyapunov exponents
"""
function lyapunov_exponent(f, u0, p, tspan, n_steps=1000, n_vectors=10, n_iterations=100)
    # Dimension of the system
    n = length(u0)
    
    # Initialize Lyapunov exponents
    lyapunov = zeros(min(n, n_vectors))
    
    # Time step
    dt = (tspan[2] - tspan[1]) / n_steps
    
    # Initialize orthogonal vectors
    Q = Matrix{Float64}(I, n, n_vectors)
    
    # Current state
    u = copy(u0)
    
    # Iterate
    for iter in 1:n_iterations
        # Evolve the system and the tangent space
        for i in 1:n_steps
            # Evolve the system
            k1 = f(u, p, 0.0)
            k2 = f(u + 0.5 * dt * k1, p, 0.0)
            k3 = f(u + 0.5 * dt * k2, p, 0.0)
            k4 = f(u + dt * k3, p, 0.0)
            u += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Evolve the tangent space
            for j in 1:n_vectors
                # Compute Jacobian-vector product using finite differences
                ε = 1e-8
                u_perturbed = u + ε * Q[:, j]
                df = (f(u_perturbed, p, 0.0) - f(u, p, 0.0)) / ε
                Q[:, j] += dt * df
            end
        end
        
        # Orthogonalize and normalize
        Q, R = qr(Q)
        
        # Update Lyapunov exponents
        for j in 1:n_vectors
            lyapunov[j] += log(abs(R[j, j])) / (dt * n_steps)
        end
    end
    
    # Average over iterations
    lyapunov ./= n_iterations
    
    return lyapunov
end

"""
    bifurcation_diagram(f, u0, p_range, p_index, tspan, n_transient=1000, n_points=1000)

Compute a bifurcation diagram for a dynamical system.

# Arguments
- `f`: Function defining the ODE (du/dt = f(u, p, t))
- `u0`: Initial condition
- `p_range`: Range of the bifurcation parameter
- `p_index`: Index of the bifurcation parameter
- `tspan`: Time span (t0, tf)
- `n_transient`: Number of transient iterations to discard
- `n_points`: Number of points to plot

# Returns
- Bifurcation diagram (parameter values and corresponding state values)
"""
function bifurcation_diagram(f, u0, p_range, p_index, tspan, n_transient=1000, n_points=1000)
    # Initialize arrays for bifurcation diagram
    p_values = []
    state_values = []
    
    # Iterate over parameter values
    for p_value in p_range
        # Update parameter
        p = copy(p_range[1])
        p[p_index] = p_value
        
        # Solve the system
        prob = ODEProblem(f, u0, tspan, p)
        sol = solve(prob, saveat=(tspan[2] - tspan[1]) / (n_transient + n_points))
        
        # Discard transient
        for i in (n_transient+1):length(sol.t)
            push!(p_values, p_value)
            push!(state_values, sol.u[i][1])  # Use first component by default
        end
        
        # Update initial condition for next parameter value
        u0 = sol.u[end]
    end
    
    return p_values, state_values
end

"""
    basin_of_attraction(f, p, tspan, x_range, y_range, n_grid=100, max_iter=1000, atol=1e-6)

Compute the basin of attraction for a 2D dynamical system.

# Arguments
- `f`: Function defining the ODE (du/dt = f(u, p, t))
- `p`: Parameters
- `tspan`: Time span (t0, tf)
- `x_range`: Range of x values
- `y_range`: Range of y values
- `n_grid`: Number of grid points in each dimension
- `max_iter`: Maximum number of iterations
- `atol`: Absolute tolerance for convergence

# Returns
- Basin of attraction (grid of attractor indices)
"""
function basin_of_attraction(f, p, tspan, x_range, y_range, n_grid=100, max_iter=1000, atol=1e-6)
    # Create grid
    x = range(x_range[1], x_range[2], length=n_grid)
    y = range(y_range[1], y_range[2], length=n_grid)
    
    # Initialize basin
    basin = zeros(Int, n_grid, n_grid)
    
    # List of attractors
    attractors = []
    
    # Iterate over grid points
    for i in 1:n_grid
        for j in 1:n_grid
            # Initial condition
            u0 = [x[i], y[j]]
            
            # Solve the system
            prob = ODEProblem(f, u0, tspan, p)
            sol = solve(prob)
            
            # Final state
            u_final = sol.u[end]
            
            # Check if the final state is close to a known attractor
            attractor_idx = 0
            for (k, attractor) in enumerate(attractors)
                if norm(u_final - attractor) < atol
                    attractor_idx = k
                    break
                end
            end
            
            # If not close to any known attractor, add a new one
            if attractor_idx == 0
                push!(attractors, u_final)
                attractor_idx = length(attractors)
            end
            
            # Assign basin
            basin[i, j] = attractor_idx
        end
    end
    
    return basin, attractors
end

"""
    fractal_dimension(points, method=:box_counting, max_size=100)

Compute the fractal dimension of a set of points.

# Arguments
- `points`: Set of points
- `method`: Method to use (:box_counting or :correlation)
- `max_size`: Maximum box size for box counting

# Returns
- Fractal dimension
"""
function fractal_dimension(points, method=:box_counting, max_size=100)
    if method == :box_counting
        # Box counting dimension
        
        # Determine the range of the points
        min_coords = minimum(points, dims=1)
        max_coords = maximum(points, dims=1)
        
        # Initialize arrays for box sizes and counts
        sizes = []
        counts = []
        
        # Iterate over box sizes
        for size in 1:max_size
            # Count boxes
            boxes = Set()
            for point in points
                # Compute box indices
                box_indices = floor.(Int, (point .- min_coords) ./ size)
                push!(boxes, tuple(box_indices...))
            end
            
            # Record size and count
            push!(sizes, size)
            push!(counts, length(boxes))
        end
        
        # Compute dimension using linear regression
        log_sizes = log.(sizes)
        log_counts = log.(counts)
        
        # Simple linear regression
        n = length(log_sizes)
        x_mean = mean(log_sizes)
        y_mean = mean(log_counts)
        
        numerator = sum((log_sizes .- x_mean) .* (log_counts .- y_mean))
        denominator = sum((log_sizes .- x_mean).^2)
        
        slope = numerator / denominator
        
        # Box counting dimension is -slope
        return -slope
    elseif method == :correlation
        # Correlation dimension
        
        # Number of points
        n = length(points)
        
        # Compute pairwise distances
        distances = []
        for i in 1:n
            for j in (i+1):n
                push!(distances, norm(points[i] - points[j]))
            end
        end
        
        # Sort distances
        sort!(distances)
        
        # Compute correlation sum
        r_values = range(minimum(distances), maximum(distances), length=100)
        c_values = []
        
        for r in r_values
            # Count pairs with distance less than r
            count = sum(distances .< r)
            
            # Correlation sum
            c = 2 * count / (n * (n - 1))
            push!(c_values, c)
        end
        
        # Compute dimension using linear regression
        log_r = log.(r_values)
        log_c = log.(c_values)
        
        # Simple linear regression
        n = length(log_r)
        x_mean = mean(log_r)
        y_mean = mean(log_c)
        
        numerator = sum((log_r .- x_mean) .* (log_c .- y_mean))
        denominator = sum((log_r .- x_mean).^2)
        
        slope = numerator / denominator
        
        # Correlation dimension is slope
        return slope
    else
        error("Unknown method: $method")
    end
end

end # module

