module NumericalMethods

using LinearAlgebra
using Statistics
using Random
using Distributions
using Optim

export integrate_trapezoid, integrate_simpson
export differentiate_central, differentiate_forward, differentiate_backward
export solve_linear_system, solve_nonlinear_system
export interpolate_linear, interpolate_cubic
export optimize_gradient_descent, optimize_newton
export monte_carlo_integration

"""
    integrate_trapezoid(f, a, b, n=1000)

Integrate function `f` from `a` to `b` using the trapezoid rule with `n` points.

# Arguments
- `f`: Function to integrate
- `a`: Lower bound
- `b`: Upper bound
- `n`: Number of points (default: 1000)

# Returns
- Approximation of the integral
"""
function integrate_trapezoid(f, a, b, n=1000)
    h = (b - a) / n
    x = a:h:b
    y = f.(x)
    return h * (sum(y) - 0.5 * (y[1] + y[end]))
end

"""
    integrate_simpson(f, a, b, n=1000)

Integrate function `f` from `a` to `b` using Simpson's rule with `n` points.

# Arguments
- `f`: Function to integrate
- `a`: Lower bound
- `b`: Upper bound
- `n`: Number of points (must be even, default: 1000)

# Returns
- Approximation of the integral
"""
function integrate_simpson(f, a, b, n=1000)
    if n % 2 != 0
        n += 1  # Ensure n is even
    end
    
    h = (b - a) / n
    x = a:h:b
    y = f.(x)
    
    return h/3 * (y[1] + y[end] + 4*sum(y[2:2:end-1]) + 2*sum(y[3:2:end-2]))
end

"""
    differentiate_central(f, x, h=1e-6)

Compute the derivative of function `f` at point `x` using central difference.

# Arguments
- `f`: Function to differentiate
- `x`: Point at which to compute the derivative
- `h`: Step size (default: 1e-6)

# Returns
- Approximation of the derivative
"""
function differentiate_central(f, x, h=1e-6)
    return (f(x + h) - f(x - h)) / (2 * h)
end

"""
    differentiate_forward(f, x, h=1e-6)

Compute the derivative of function `f` at point `x` using forward difference.

# Arguments
- `f`: Function to differentiate
- `x`: Point at which to compute the derivative
- `h`: Step size (default: 1e-6)

# Returns
- Approximation of the derivative
"""
function differentiate_forward(f, x, h=1e-6)
    return (f(x + h) - f(x)) / h
end

"""
    differentiate_backward(f, x, h=1e-6)

Compute the derivative of function `f` at point `x` using backward difference.

# Arguments
- `f`: Function to differentiate
- `x`: Point at which to compute the derivative
- `h`: Step size (default: 1e-6)

# Returns
- Approximation of the derivative
"""
function differentiate_backward(f, x, h=1e-6)
    return (f(x) - f(x - h)) / h
end

"""
    solve_linear_system(A, b)

Solve the linear system Ax = b.

# Arguments
- `A`: Coefficient matrix
- `b`: Right-hand side vector

# Returns
- Solution vector x
"""
function solve_linear_system(A, b)
    return A \ b
end

"""
    solve_nonlinear_system(f, x0; tol=1e-8, max_iter=1000)

Solve a system of nonlinear equations using Newton's method.

# Arguments
- `f`: Function that returns the system of equations
- `x0`: Initial guess
- `tol`: Tolerance for convergence (default: 1e-8)
- `max_iter`: Maximum number of iterations (default: 1000)

# Returns
- Solution vector
"""
function solve_nonlinear_system(f, x0; tol=1e-8, max_iter=1000)
    x = copy(x0)
    
    for i in 1:max_iter
        fx = f(x)
        if norm(fx) < tol
            return x
        end
        
        # Compute Jacobian using finite differences
        n = length(x)
        J = zeros(n, n)
        h = 1e-8
        
        for j in 1:n
            x_plus_h = copy(x)
            x_plus_h[j] += h
            J[:, j] = (f(x_plus_h) - fx) / h
        end
        
        # Newton step
        dx = J \ (-fx)
        x += dx
        
        if norm(dx) < tol
            return x
        end
    end
    
    error("Failed to converge after $(max_iter) iterations")
end

"""
    interpolate_linear(x, y, x_new)

Perform linear interpolation.

# Arguments
- `x`: x-coordinates of data points
- `y`: y-coordinates of data points
- `x_new`: x-coordinates at which to interpolate

# Returns
- Interpolated y-coordinates
"""
function interpolate_linear(x, y, x_new)
    n = length(x)
    y_new = similar(x_new)
    
    for (i, xi) in enumerate(x_new)
        # Find the interval containing xi
        idx = findfirst(j -> x[j] >= xi, 1:n)
        
        if isnothing(idx) || idx == 1
            if isnothing(idx)
                # Extrapolate using the last interval
                idx = n
                j1, j2 = n-1, n
            elseif idx == 1
                # Extrapolate using the first interval
                j1, j2 = 1, 2
            end
        else
            j1, j2 = idx-1, idx
        end
        
        # Linear interpolation formula
        t = (xi - x[j1]) / (x[j2] - x[j1])
        y_new[i] = (1 - t) * y[j1] + t * y[j2]
    end
    
    return y_new
end

"""
    interpolate_cubic(x, y, x_new)

Perform cubic spline interpolation.

# Arguments
- `x`: x-coordinates of data points
- `y`: y-coordinates of data points
- `x_new`: x-coordinates at which to interpolate

# Returns
- Interpolated y-coordinates
"""
function interpolate_cubic(x, y, x_new)
    n = length(x)
    
    # Compute second derivatives
    h = diff(x)
    α = 3 ./ h[1:end-1] .* diff(y)[2:end] - 3 ./ h[2:end] .* diff(y)[1:end-1]
    
    # Tridiagonal system
    l = zeros(n)
    μ = zeros(n)
    z = zeros(n)
    
    l[1] = 1
    μ[n] = 1
    
    for i in 2:n-1
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * μ[i-1]
        μ[i] = h[i] / l[i]
        z[i] = (α[i-1] - h[i-1] * z[i-1]) / l[i]
    end
    
    # Back-substitution
    c = zeros(n)
    b = zeros(n-1)
    d = zeros(n-1)
    
    for j in n-1:-1:1
        c[j] = z[j] - μ[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
    end
    
    # Interpolate
    y_new = similar(x_new)
    
    for (i, xi) in enumerate(x_new)
        # Find the interval containing xi
        idx = findfirst(j -> x[j] > xi, 1:n)
        
        if isnothing(idx)
            # Extrapolate using the last interval
            idx = n
        end
        
        if idx == 1
            # Extrapolate using the first interval
            j = 1
        else
            j = idx - 1
        end
        
        # Cubic interpolation formula
        dx = xi - x[j]
        y_new[i] = y[j] + b[j] * dx + c[j] * dx^2 + d[j] * dx^3
    end
    
    return y_new
end

"""
    optimize_gradient_descent(f, g, x0; α=0.01, tol=1e-6, max_iter=1000)

Minimize function `f` using gradient descent.

# Arguments
- `f`: Function to minimize
- `g`: Gradient of the function
- `x0`: Initial guess
- `α`: Learning rate (default: 0.01)
- `tol`: Tolerance for convergence (default: 1e-6)
- `max_iter`: Maximum number of iterations (default: 1000)

# Returns
- Minimizer
- Minimum value
- Number of iterations
"""
function optimize_gradient_descent(f, g, x0; α=0.01, tol=1e-6, max_iter=1000)
    x = copy(x0)
    fx = f(x)
    
    for i in 1:max_iter
        gx = g(x)
        
        if norm(gx) < tol
            return x, fx, i
        end
        
        x_new = x - α * gx
        fx_new = f(x_new)
        
        if abs(fx_new - fx) < tol
            return x_new, fx_new, i
        end
        
        x = x_new
        fx = fx_new
    end
    
    return x, fx, max_iter
end

"""
    optimize_newton(f, g, h, x0; tol=1e-6, max_iter=100)

Minimize function `f` using Newton's method.

# Arguments
- `f`: Function to minimize
- `g`: Gradient of the function
- `h`: Hessian of the function
- `x0`: Initial guess
- `tol`: Tolerance for convergence (default: 1e-6)
- `max_iter`: Maximum number of iterations (default: 100)

# Returns
- Minimizer
- Minimum value
- Number of iterations
"""
function optimize_newton(f, g, h, x0; tol=1e-6, max_iter=100)
    x = copy(x0)
    fx = f(x)
    
    for i in 1:max_iter
        gx = g(x)
        
        if norm(gx) < tol
            return x, fx, i
        end
        
        hx = h(x)
        dx = hx \ (-gx)
        
        x_new = x + dx
        fx_new = f(x_new)
        
        if abs(fx_new - fx) < tol
            return x_new, fx_new, i
        end
        
        x = x_new
        fx = fx_new
    end
    
    return x, fx, max_iter
end

"""
    monte_carlo_integration(f, a, b, n=10000)

Integrate function `f` from `a` to `b` using Monte Carlo integration with `n` points.

# Arguments
- `f`: Function to integrate
- `a`: Lower bounds (vector for multi-dimensional integration)
- `b`: Upper bounds (vector for multi-dimensional integration)
- `n`: Number of points (default: 10000)

# Returns
- Approximation of the integral
- Estimated error
"""
function monte_carlo_integration(f, a, b, n=10000)
    if isa(a, Number)
        a = [a]
        b = [b]
    end
    
    d = length(a)
    volume = prod(b - a)
    
    # Generate random points
    points = zeros(n, d)
    for i in 1:d
        points[:, i] = a[i] .+ (b[i] - a[i]) .* rand(n)
    end
    
    # Evaluate function at each point
    values = zeros(n)
    for i in 1:n
        values[i] = f(points[i, :])
    end
    
    # Compute integral and error estimate
    integral = volume * mean(values)
    error = volume * std(values) / sqrt(n)
    
    return integral, error
end

end # module

