# 🇧🇷 Computação Científica com Julia | 🇺🇸 Scientific Computing with Julia

<div align="center">


**Plataforma avançada de computação científica com Julia para análise numérica, simulações e computação de alta performance**

[🧮 Algoritmos](#-algoritmos-implementados) • [⚡ Performance](#-benchmarks-de-performance) • [🔬 Simulações](#-simulações-científicas) • [🚀 Setup](#-setup-rápido)

</div>

---

## 🇧🇷 Português

### 🧮 Visão Geral

Plataforma abrangente de **computação científica** desenvolvida em Julia para análise numérica de alta performance:

- 🔢 **Álgebra Linear**: Operações matriciais otimizadas e decomposições
- 📊 **Análise Numérica**: Métodos numéricos avançados e precisão dupla
- 🌊 **Equações Diferenciais**: Solvers para EDOs, EDPs e sistemas dinâmicos
- 🎯 **Otimização**: Algoritmos de otimização linear e não-linear
- 🔬 **Simulações**: Monte Carlo, dinâmica molecular, física computacional
- ⚡ **Computação Paralela**: GPU computing e processamento distribuído

### 🎯 Objetivos da Plataforma

- **Acelerar computações** científicas com performance próxima ao C
- **Implementar algoritmos** numéricos state-of-the-art
- **Facilitar simulações** complexas em física e engenharia
- **Otimizar problemas** de grande escala com métodos avançados
- **Democratizar HPC** com interface amigável e documentação clara

### 🛠️ Stack Tecnológico

#### Core Julia
- **Julia 1.9+**: Linguagem principal para computação científica
- **LinearAlgebra.jl**: Álgebra linear de alta performance
- **DifferentialEquations.jl**: Solvers para equações diferenciais
- **Optimization.jl**: Framework unificado de otimização

#### Computação Numérica
- **BLAS/LAPACK**: Bibliotecas otimizadas de álgebra linear
- **FFTW.jl**: Transformadas de Fourier rápidas
- **QuadGK.jl**: Integração numérica adaptativa
- **Roots.jl**: Encontrar raízes de funções

#### Computação Paralela e GPU
- **CUDA.jl**: Computação em GPU NVIDIA
- **MPI.jl**: Computação distribuída
- **Threads.jl**: Paralelização multi-thread
- **Distributed.jl**: Computação distribuída

#### Visualização Científica
- **Plots.jl**: Visualizações científicas
- **PlotlyJS.jl**: Gráficos interativos
- **Makie.jl**: Visualizações 3D avançadas
- **PyPlot.jl**: Interface para matplotlib

#### Simulações e Modelagem
- **DynamicalSystems.jl**: Sistemas dinâmicos
- **StochasticDiffEq.jl**: Equações diferenciais estocásticas
- **Catalyst.jl**: Modelagem de redes de reações
- **ModelingToolkit.jl**: Modelagem simbólica

### 📋 Estrutura da Plataforma

```
julia-scientific-computing/
├── 📁 src/                        # Código fonte principal
│   ├── 📁 linear_algebra/         # Álgebra linear avançada
│   │   ├── 📄 matrix_operations.jl # Operações matriciais
│   │   ├── 📄 decompositions.jl   # Decomposições (SVD, QR, LU)
│   │   ├── 📄 eigenvalue_problems.jl # Problemas de autovalores
│   │   ├── 📄 sparse_matrices.jl  # Matrizes esparsas
│   │   └── 📄 iterative_solvers.jl # Solvers iterativos
│   ├── 📁 numerical_analysis/     # Análise numérica
│   │   ├── 📄 interpolation.jl    # Interpolação e aproximação
│   │   ├── 📄 integration.jl      # Integração numérica
│   │   ├── 📄 differentiation.jl  # Diferenciação numérica
│   │   ├── 📄 root_finding.jl     # Encontrar raízes
│   │   └── 📄 fourier_analysis.jl # Análise de Fourier
│   ├── 📁 differential_equations/ # Equações diferenciais
│   │   ├── 📄 ode_solvers.jl      # Solvers para EDOs
│   │   ├── 📄 pde_solvers.jl      # Solvers para EDPs
│   │   ├── 📄 stochastic_de.jl    # Equações estocásticas
│   │   ├── 📄 delay_equations.jl  # Equações com atraso
│   │   └── 📄 boundary_problems.jl # Problemas de contorno
│   ├── 📁 optimization/           # Otimização
│   │   ├── 📄 linear_programming.jl # Programação linear
│   │   ├── 📄 nonlinear_optimization.jl # Otimização não-linear
│   │   ├── 📄 global_optimization.jl # Otimização global
│   │   ├── 📄 constrained_optimization.jl # Otimização restrita
│   │   └── 📄 metaheuristics.jl   # Algoritmos metaheurísticos
│   ├── 📁 simulations/            # Simulações científicas
│   │   ├── 📄 monte_carlo.jl      # Simulações Monte Carlo
│   │   ├── 📄 molecular_dynamics.jl # Dinâmica molecular
│   │   ├── 📄 fluid_dynamics.jl   # Dinâmica de fluidos
│   │   ├── 📄 quantum_mechanics.jl # Mecânica quântica
│   │   └── 📄 statistical_mechanics.jl # Mecânica estatística
│   ├── 📁 parallel_computing/     # Computação paralela
│   │   ├── 📄 gpu_computing.jl    # Computação em GPU
│   │   ├── 📄 distributed_computing.jl # Computação distribuída
│   │   ├── 📄 multithreading.jl   # Multi-threading
│   │   └── 📄 cluster_computing.jl # Computação em cluster
│   ├── 📁 signal_processing/      # Processamento de sinais
│   │   ├── 📄 digital_filters.jl  # Filtros digitais
│   │   ├── 📄 spectral_analysis.jl # Análise espectral
│   │   ├── 📄 wavelets.jl         # Transformadas wavelet
│   │   └── 📄 time_series.jl      # Análise de séries temporais
│   ├── 📁 machine_learning/       # ML científico
│   │   ├── 📄 neural_odes.jl      # Neural ODEs
│   │   ├── 📄 physics_informed_nn.jl # Physics-informed NNs
│   │   ├── 📄 gaussian_processes.jl # Processos gaussianos
│   │   └── 📄 bayesian_inference.jl # Inferência bayesiana
│   └── 📁 utils/                  # Utilitários
│       ├── 📄 benchmarking.jl     # Benchmarks de performance
│       ├── 📄 visualization.jl    # Utilitários de visualização
│       ├── 📄 data_io.jl          # Input/output de dados
│       └── 📄 testing_utils.jl    # Utilitários de teste
├── 📁 examples/                   # Exemplos práticos
│   ├── 📁 physics/                # Exemplos de física
│   │   ├── 📄 pendulum_simulation.jl # Simulação de pêndulo
│   │   ├── 📄 wave_equation.jl    # Equação da onda
│   │   ├── 📄 heat_equation.jl    # Equação do calor
│   │   └── 📄 schrodinger_equation.jl # Equação de Schrödinger
│   ├── 📁 engineering/            # Exemplos de engenharia
│   │   ├── 📄 structural_analysis.jl # Análise estrutural
│   │   ├── 📄 control_systems.jl  # Sistemas de controle
│   │   ├── 📄 signal_processing.jl # Processamento de sinais
│   │   └── 📄 optimization_problems.jl # Problemas de otimização
│   ├── 📁 finance/                # Finanças quantitativas
│   │   ├── 📄 option_pricing.jl   # Precificação de opções
│   │   ├── 📄 portfolio_optimization.jl # Otimização de portfólio
│   │   ├── 📄 risk_analysis.jl    # Análise de risco
│   │   └── 📄 monte_carlo_finance.jl # Monte Carlo financeiro
│   └── 📁 biology/                # Biologia computacional
│       ├── 📄 population_dynamics.jl # Dinâmica populacional
│       ├── 📄 epidemiology.jl     # Modelos epidemiológicos
│       ├── 📄 protein_folding.jl  # Dobramento de proteínas
│       └── 📄 gene_networks.jl    # Redes gênicas
├── 📁 notebooks/                  # Jupyter notebooks
│   ├── 📄 01_linear_algebra_tutorial.ipynb # Tutorial álgebra linear
│   ├── 📄 02_differential_equations.ipynb # Equações diferenciais
│   ├── 📄 03_optimization_methods.ipynb # Métodos de otimização
│   ├── 📄 04_monte_carlo_methods.ipynb # Métodos Monte Carlo
│   ├── 📄 05_gpu_computing.ipynb  # Computação em GPU
│   ├── 📄 06_parallel_algorithms.ipynb # Algoritmos paralelos
│   ├── 📄 07_scientific_ml.ipynb  # Machine learning científico
│   └── 📄 08_performance_optimization.ipynb # Otimização de performance
├── 📁 benchmarks/                 # Benchmarks de performance
│   ├── 📄 linear_algebra_bench.jl # Benchmark álgebra linear
│   ├── 📄 ode_solvers_bench.jl    # Benchmark solvers EDO
│   ├── 📄 optimization_bench.jl   # Benchmark otimização
│   ├── 📄 gpu_vs_cpu_bench.jl     # Comparação GPU vs CPU
│   └── 📄 julia_vs_others.jl      # Julia vs outras linguagens
├── 📁 data/                       # Dados para exemplos
│   ├── 📁 experimental/           # Dados experimentais
│   ├── 📁 synthetic/              # Dados sintéticos
│   ├── 📁 reference/              # Dados de referência
│   └── 📁 benchmarks/             # Dados para benchmarks
├── 📁 docs/                       # Documentação
│   ├── 📄 theory/                 # Fundamentação teórica
│   ├── 📄 algorithms/             # Descrição de algoritmos
│   ├── 📄 performance/            # Análise de performance
│   └── 📄 tutorials/              # Tutoriais detalhados
├── 📁 tests/                      # Testes automatizados
│   ├── 📄 test_linear_algebra.jl  # Testes álgebra linear
│   ├── 📄 test_numerical_methods.jl # Testes métodos numéricos
│   ├── 📄 test_optimization.jl    # Testes otimização
│   └── 📄 test_simulations.jl     # Testes simulações
├── 📄 Project.toml                # Dependências Julia
├── 📄 Manifest.toml               # Lock file de dependências
├── 📄 README.md                   # Este arquivo
├── 📄 LICENSE                     # Licença MIT
└── 📄 .gitignore                 # Arquivos ignorados
```

### 🧮 Algoritmos Implementados

#### 1. 🔢 Álgebra Linear Avançada

**Decomposições Matriciais Otimizadas**
```julia
module LinearAlgebraAdvanced

using LinearAlgebra, SparseArrays, CUDA

"""
Decomposição SVD otimizada para matrizes grandes
"""
function optimized_svd(A::AbstractMatrix{T}; 
                      rank_threshold::Real = 1e-12,
                      use_gpu::Bool = false) where T
    
    if use_gpu && CUDA.functional()
        A_gpu = CuArray(A)
        U, S, V = svd(A_gpu)
        
        # Filtrar valores singulares pequenos
        significant_indices = S .> rank_threshold
        U_reduced = U[:, significant_indices]
        S_reduced = S[significant_indices]
        V_reduced = V[:, significant_indices]
        
        return Array(U_reduced), Array(S_reduced), Array(V_reduced)
    else
        U, S, V = svd(A)
        significant_indices = S .> rank_threshold
        return U[:, significant_indices], S[significant_indices], V[:, significant_indices]
    end
end

"""
Solver iterativo para sistemas lineares grandes e esparsos
"""
function iterative_solve(A::SparseMatrixCSC{T}, b::Vector{T};
                        method::Symbol = :gmres,
                        tol::Real = 1e-10,
                        maxiter::Int = 1000) where T
    
    n = size(A, 1)
    x = zeros(T, n)
    
    if method == :gmres
        # Implementação GMRES
        return gmres_solver(A, b, x, tol, maxiter)
    elseif method == :cg
        # Gradiente Conjugado (para matrizes simétricas positivas definidas)
        return conjugate_gradient(A, b, x, tol, maxiter)
    elseif method == :bicgstab
        # BiCGSTAB para matrizes não-simétricas
        return bicgstab_solver(A, b, x, tol, maxiter)
    else
        error("Método não suportado: $method")
    end
end

"""
Implementação otimizada do algoritmo GMRES
"""
function gmres_solver(A, b, x0, tol, maxiter)
    n = length(b)
    m = min(maxiter, n)
    
    # Inicialização
    r0 = b - A * x0
    β = norm(r0)
    
    if β < tol
        return x0, 0, true
    end
    
    # Base ortonormal de Krylov
    V = zeros(eltype(b), n, m + 1)
    V[:, 1] = r0 / β
    
    # Matriz de Hessenberg superior
    H = zeros(eltype(b), m + 1, m)
    
    # Vetor para o problema de mínimos quadrados
    g = zeros(eltype(b), m + 1)
    g[1] = β
    
    for j = 1:m
        # Produto matriz-vetor
        w = A * V[:, j]
        
        # Processo de Gram-Schmidt modificado
        for i = 1:j
            H[i, j] = dot(w, V[:, i])
            w -= H[i, j] * V[:, i]
        end
        
        H[j + 1, j] = norm(w)
        
        if H[j + 1, j] < tol
            # Convergência prematura
            m = j
            break
        end
        
        V[:, j + 1] = w / H[j + 1, j]
        
        # Resolver problema de mínimos quadrados
        y = H[1:j+1, 1:j] \ g[1:j+1]
        
        # Verificar convergência
        residual_norm = abs(g[j + 1] - H[j + 1, j] * y[j])
        
        if residual_norm < tol
            x = x0 + V[:, 1:j] * y
            return x, j, true
        end
    end
    
    # Solução final
    y = H[1:m+1, 1:m] \ g[1:m+1]
    x = x0 + V[:, 1:m] * y
    
    return x, m, norm(b - A * x) < tol
end

"""
Decomposição QR com pivoteamento para estabilidade numérica
"""
function qr_pivoted_stable(A::AbstractMatrix{T}) where T
    m, n = size(A)
    
    # Inicialização
    Q = Matrix{T}(I, m, m)
    R = copy(A)
    P = collect(1:n)
    
    for k = 1:min(m-1, n)
        # Escolher pivô (coluna com maior norma)
        norms = [norm(R[k:end, j]) for j in k:n]
        pivot_idx = argmax(norms) + k - 1
        
        if pivot_idx != k
            # Trocar colunas
            R[:, [k, pivot_idx]] = R[:, [pivot_idx, k]]
            P[k], P[pivot_idx] = P[pivot_idx], P[k]
        end
        
        # Reflexão de Householder
        x = R[k:end, k]
        v = copy(x)
        v[1] += sign(x[1]) * norm(x)
        v = v / norm(v)
        
        # Aplicar reflexão
        R[k:end, k:end] -= 2 * v * (v' * R[k:end, k:end])
        Q[:, k:end] -= 2 * (Q[:, k:end] * v) * v'
    end
    
    return Q, R, P
end

end # module
```

**Eigenvalue Problems para Matrizes Grandes**
```julia
"""
Algoritmo de Lanczos para problemas de autovalores de matrizes simétricas grandes
"""
function lanczos_eigenvalues(A::AbstractMatrix{T}, 
                           k::Int = 6;
                           tol::Real = 1e-12,
                           maxiter::Int = 300) where T
    
    n = size(A, 1)
    @assert issymmetric(A) "Matriz deve ser simétrica para o algoritmo de Lanczos"
    
    # Inicialização
    q = randn(T, n)
    q = q / norm(q)
    
    # Matrizes de Lanczos
    Q = zeros(T, n, min(k + 20, n))  # Buffer extra para convergência
    α = zeros(T, min(k + 20, n))
    β = zeros(T, min(k + 20, n))
    
    Q[:, 1] = q
    
    for j = 1:min(k + 20, n) - 1
        # Produto matriz-vetor
        v = A * Q[:, j]
        
        # Ortogonalização
        α[j] = dot(Q[:, j], v)
        v -= α[j] * Q[:, j]
        
        if j > 1
            v -= β[j-1] * Q[:, j-1]
        end
        
        # Re-ortogonalização (para estabilidade numérica)
        for i = 1:j
            proj = dot(Q[:, i], v)
            v -= proj * Q[:, i]
        end
        
        β[j] = norm(v)
        
        if β[j] < tol
            # Convergência
            break
        end
        
        Q[:, j+1] = v / β[j]
        
        # Verificar convergência dos autovalores
        if j >= k
            T_matrix = SymTridiagonal(α[1:j], β[1:j-1])
            eigenvals = eigvals(T_matrix)
            
            # Critério de convergência baseado na estabilidade dos autovalores
            if j > k + 5
                T_prev = SymTridiagonal(α[1:j-1], β[1:j-2])
                eigenvals_prev = eigvals(T_prev)
                
                # Verificar se os k maiores autovalores convergiram
                if maximum(abs.(eigenvals[end-k+1:end] - eigenvals_prev[end-k+1:end])) < tol
                    break
                end
            end
        end
    end
    
    # Construir matriz tridiagonal final
    m = min(j, size(Q, 2))
    T_matrix = SymTridiagonal(α[1:m], β[1:m-1])
    
    # Calcular autovalores e autovetores
    eigenvals, eigenvecs_T = eigen(T_matrix)
    
    # Transformar autovetores de volta ao espaço original
    eigenvecs = Q[:, 1:m] * eigenvecs_T
    
    # Retornar os k maiores autovalores
    perm = sortperm(eigenvals, rev=true)
    return eigenvals[perm[1:k]], eigenvecs[:, perm[1:k]]
end
```

#### 2. 🌊 Equações Diferenciais Avançadas

**Solver Adaptativo para EDOs Stiff**
```julia
module DifferentialEquationsSolvers

using DifferentialEquations, LinearAlgebra

"""
Solver Rosenbrock para sistemas stiff de EDOs
"""
function rosenbrock_solver(f, jac, u0, tspan; 
                          dt_initial = 1e-3,
                          rtol = 1e-6,
                          atol = 1e-9)
    
    t0, tf = tspan
    u = copy(u0)
    t = t0
    dt = dt_initial
    
    # Armazenar solução
    solution_t = [t0]
    solution_u = [copy(u0)]
    
    # Parâmetros do método Rosenbrock
    γ = 1.0 / (2.0 + sqrt(2.0))
    a21 = 1.0 / γ
    c2 = 1.0
    
    while t < tf
        # Ajustar passo se necessário
        if t + dt > tf
            dt = tf - t
        end
        
        # Jacobiano no ponto atual
        J = jac(u, nothing, t)
        
        # Matriz do sistema linear
        W = I - γ * dt * J
        
        # Estágios do método Rosenbrock
        k1 = W \ f(u, nothing, t)
        k2 = W \ (f(u + a21 * dt * k1, nothing, t + c2 * dt) - 2 * γ * dt * J * k1)
        
        # Nova solução
        u_new = u + dt * (k1 + k2) / 2
        
        # Estimativa do erro
        error_est = dt * abs(k2 - k1) / 2
        error_norm = maximum(error_est ./ (atol .+ rtol .* max.(abs.(u), abs.(u_new))))
        
        if error_norm <= 1.0
            # Aceitar passo
            u = u_new
            t += dt
            push!(solution_t, t)
            push!(solution_u, copy(u))
            
            # Ajustar passo para próxima iteração
            dt *= min(2.0, 0.9 * (1.0 / error_norm)^(1/3))
        else
            # Rejeitar passo e diminuir dt
            dt *= max(0.1, 0.9 * (1.0 / error_norm)^(1/3))
        end
        
        # Limitar tamanho do passo
        dt = min(dt, 0.1)
        dt = max(dt, 1e-12)
    end
    
    return solution_t, solution_u
end

"""
Solver para EDPs usando método de diferenças finitas
"""
function pde_heat_equation_2d(nx::Int, ny::Int, nt::Int,
                             Lx::Real, Ly::Real, T::Real;
                             α::Real = 1.0,
                             initial_condition = (x, y) -> sin(π*x/Lx) * sin(π*y/Ly),
                             boundary_condition = (x, y, t) -> 0.0)
    
    # Discretização espacial
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T / nt
    
    # Verificar estabilidade (critério CFL)
    cfl = α * dt * (1/dx^2 + 1/dy^2)
    @assert cfl <= 0.5 "Critério CFL violado: CFL = $cfl > 0.5"
    
    # Grids
    x = range(0, Lx, length=nx)
    y = range(0, Ly, length=ny)
    
    # Condição inicial
    u = zeros(nx, ny, nt+1)
    for i = 1:nx, j = 1:ny
        u[i, j, 1] = initial_condition(x[i], y[j])
    end
    
    # Coeficientes para diferenças finitas
    rx = α * dt / dx^2
    ry = α * dt / dy^2
    
    # Loop temporal
    for n = 1:nt
        for i = 2:nx-1, j = 2:ny-1
            u[i, j, n+1] = u[i, j, n] + 
                          rx * (u[i+1, j, n] - 2*u[i, j, n] + u[i-1, j, n]) +
                          ry * (u[i, j+1, n] - 2*u[i, j, n] + u[i, j-1, n])
        end
        
        # Aplicar condições de contorno
        t_current = n * dt
        for i = 1:nx
            u[i, 1, n+1] = boundary_condition(x[i], y[1], t_current)
            u[i, ny, n+1] = boundary_condition(x[i], y[ny], t_current)
        end
        for j = 1:ny
            u[1, j, n+1] = boundary_condition(x[1], y[j], t_current)
            u[nx, j, n+1] = boundary_condition(x[nx], y[j], t_current)
        end
    end
    
    return x, y, range(0, T, length=nt+1), u
end

end # module
```

#### 3. 🎯 Otimização Avançada

**Algoritmos de Otimização Global**
```julia
module GlobalOptimization

using Random, LinearAlgebra, Statistics

"""
Algoritmo Genético para otimização global
"""
mutable struct GeneticAlgorithm
    population_size::Int
    mutation_rate::Float64
    crossover_rate::Float64
    elitism_rate::Float64
    max_generations::Int
    
    function GeneticAlgorithm(;pop_size=100, mut_rate=0.1, cross_rate=0.8, 
                             elite_rate=0.1, max_gen=1000)
        new(pop_size, mut_rate, cross_rate, elite_rate, max_gen)
    end
end

function optimize(ga::GeneticAlgorithm, objective_function, bounds;
                 minimize=true, seed=42)
    
    Random.seed!(seed)
    
    n_vars = length(bounds)
    pop_size = ga.population_size
    
    # Inicializar população
    population = initialize_population(pop_size, bounds)
    
    # Avaliar população inicial
    fitness = [objective_function(ind) for ind in population]
    
    # Histórico de convergência
    best_fitness_history = Float64[]
    mean_fitness_history = Float64[]
    
    for generation = 1:ga.max_generations
        # Seleção por torneio
        parents = tournament_selection(population, fitness, minimize)
        
        # Crossover e mutação
        offspring = reproduce(parents, ga, bounds)
        
        # Avaliar offspring
        offspring_fitness = [objective_function(ind) for ind in offspring]
        
        # Seleção de sobreviventes (elitismo)
        population, fitness = survivor_selection(
            population, fitness, offspring, offspring_fitness, 
            ga.elitism_rate, minimize
        )
        
        # Estatísticas
        best_fit = minimize ? minimum(fitness) : maximum(fitness)
        mean_fit = mean(fitness)
        
        push!(best_fitness_history, best_fit)
        push!(mean_fitness_history, mean_fit)
        
        # Critério de parada
        if generation > 50
            recent_improvement = abs(best_fitness_history[end] - best_fitness_history[end-50])
            if recent_improvement < 1e-10
                println("Convergência atingida na geração $generation")
                break
            end
        end
    end
    
    # Melhor solução
    best_idx = minimize ? argmin(fitness) : argmax(fitness)
    best_solution = population[best_idx]
    best_value = fitness[best_idx]
    
    return (
        solution = best_solution,
        objective_value = best_value,
        convergence_history = (best_fitness_history, mean_fitness_history)
    )
end

function initialize_population(pop_size, bounds)
    n_vars = length(bounds)
    population = Vector{Vector{Float64}}(undef, pop_size)
    
    for i = 1:pop_size
        individual = zeros(n_vars)
        for j = 1:n_vars
            lower, upper = bounds[j]
            individual[j] = lower + rand() * (upper - lower)
        end
        population[i] = individual
    end
    
    return population
end

function tournament_selection(population, fitness, minimize; tournament_size=3)
    pop_size = length(population)
    parents = Vector{Vector{Float64}}(undef, pop_size)
    
    for i = 1:pop_size
        # Selecionar candidatos aleatórios
        candidates = sample(1:pop_size, tournament_size, replace=false)
        
        # Encontrar melhor candidato
        if minimize
            winner_idx = candidates[argmin(fitness[candidates])]
        else
            winner_idx = candidates[argmax(fitness[candidates])]
        end
        
        parents[i] = copy(population[winner_idx])
    end
    
    return parents
end

function reproduce(parents, ga, bounds)
    pop_size = length(parents)
    offspring = Vector{Vector{Float64}}(undef, pop_size)
    
    for i = 1:2:pop_size-1
        parent1 = parents[i]
        parent2 = parents[i+1]
        
        # Crossover
        if rand() < ga.crossover_rate
            child1, child2 = simulated_binary_crossover(parent1, parent2, bounds)
        else
            child1, child2 = copy(parent1), copy(parent2)
        end
        
        # Mutação
        if rand() < ga.mutation_rate
            polynomial_mutation!(child1, bounds)
        end
        if rand() < ga.mutation_rate
            polynomial_mutation!(child2, bounds)
        end
        
        offspring[i] = child1
        if i+1 <= pop_size
            offspring[i+1] = child2
        end
    end
    
    return offspring
end

"""
Particle Swarm Optimization (PSO)
"""
mutable struct ParticleSwarmOptimizer
    n_particles::Int
    max_iterations::Int
    w::Float64  # inertia weight
    c1::Float64  # cognitive parameter
    c2::Float64  # social parameter
    
    function ParticleSwarmOptimizer(;n_particles=30, max_iter=1000, 
                                   w=0.7, c1=2.0, c2=2.0)
        new(n_particles, max_iter, w, c1, c2)
    end
end

function optimize(pso::ParticleSwarmOptimizer, objective_function, bounds;
                 minimize=true, seed=42)
    
    Random.seed!(seed)
    
    n_vars = length(bounds)
    n_particles = pso.n_particles
    
    # Inicializar partículas
    positions = [initialize_particle(bounds) for _ = 1:n_particles]
    velocities = [zeros(n_vars) for _ = 1:n_particles]
    
    # Avaliar posições iniciais
    fitness = [objective_function(pos) for pos in positions]
    
    # Melhores posições pessoais
    personal_best_positions = copy(positions)
    personal_best_fitness = copy(fitness)
    
    # Melhor posição global
    global_best_idx = minimize ? argmin(fitness) : argmax(fitness)
    global_best_position = copy(positions[global_best_idx])
    global_best_fitness = fitness[global_best_idx]
    
    # Histórico de convergência
    convergence_history = [global_best_fitness]
    
    for iteration = 1:pso.max_iterations
        for i = 1:n_particles
            # Atualizar velocidade
            r1, r2 = rand(n_vars), rand(n_vars)
            
            velocities[i] = pso.w * velocities[i] +
                           pso.c1 * r1 .* (personal_best_positions[i] - positions[i]) +
                           pso.c2 * r2 .* (global_best_position - positions[i])
            
            # Atualizar posição
            positions[i] += velocities[i]
            
            # Aplicar limites
            for j = 1:n_vars
                lower, upper = bounds[j]
                positions[i][j] = clamp(positions[i][j], lower, upper)
            end
            
            # Avaliar nova posição
            new_fitness = objective_function(positions[i])
            
            # Atualizar melhor pessoal
            if (minimize && new_fitness < personal_best_fitness[i]) ||
               (!minimize && new_fitness > personal_best_fitness[i])
                personal_best_positions[i] = copy(positions[i])
                personal_best_fitness[i] = new_fitness
                
                # Atualizar melhor global
                if (minimize && new_fitness < global_best_fitness) ||
                   (!minimize && new_fitness > global_best_fitness)
                    global_best_position = copy(positions[i])
                    global_best_fitness = new_fitness
                end
            end
        end
        
        push!(convergence_history, global_best_fitness)
        
        # Critério de parada
        if iteration > 100
            recent_improvement = abs(convergence_history[end] - convergence_history[end-100])
            if recent_improvement < 1e-12
                println("PSO convergiu na iteração $iteration")
                break
            end
        end
    end
    
    return (
        solution = global_best_position,
        objective_value = global_best_fitness,
        convergence_history = convergence_history
    )
end

function initialize_particle(bounds)
    n_vars = length(bounds)
    particle = zeros(n_vars)
    
    for i = 1:n_vars
        lower, upper = bounds[i]
        particle[i] = lower + rand() * (upper - lower)
    end
    
    return particle
end

end # module
```

### ⚡ Benchmarks de Performance

**Comparação Julia vs Outras Linguagens**
```julia
module PerformanceBenchmarks

using BenchmarkTools, LinearAlgebra, Random

"""
Benchmark de multiplicação de matrizes
"""
function benchmark_matrix_multiplication(sizes = [100, 500, 1000, 2000])
    results = Dict()
    
    for n in sizes
        println("Benchmarking matrix multiplication for size $n x $n")
        
        # Gerar matrizes aleatórias
        A = randn(n, n)
        B = randn(n, n)
        
        # Benchmark
        benchmark_result = @benchmark $A * $B
        
        results[n] = (
            median_time = median(benchmark_result.times) / 1e9,  # em segundos
            memory = benchmark_result.memory,
            gflops = 2 * n^3 / (median(benchmark_result.times) / 1e9) / 1e9
        )
        
        println("  Tempo mediano: $(results[n].median_time) s")
        println("  GFLOPS: $(results[n].gflops)")
        println("  Memória: $(results[n].memory) bytes")
    end
    
    return results
end

"""
Benchmark de solvers de equações diferenciais
"""
function benchmark_ode_solvers()
    using DifferentialEquations
    
    # Problema de teste: Oscilador harmônico
    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end
    
    u0 = [1.0, 0.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(harmonic_oscillator!, u0, tspan)
    
    # Diferentes solvers
    solvers = [
        ("Tsit5", Tsit5()),
        ("Vern7", Vern7()),
        ("DormandPrince", DP5()),
        ("RadauIIA5", RadauIIA5())
    ]
    
    results = Dict()
    
    for (name, solver) in solvers
        println("Benchmarking ODE solver: $name")
        
        benchmark_result = @benchmark solve($prob, $solver, reltol=1e-8, abstol=1e-10)
        
        results[name] = (
            median_time = median(benchmark_result.times) / 1e6,  # em ms
            memory = benchmark_result.memory,
            allocations = benchmark_result.allocs
        )
        
        println("  Tempo mediano: $(results[name].median_time) ms")
        println("  Memória: $(results[name].memory) bytes")
        println("  Alocações: $(results[name].allocations)")
    end
    
    return results
end

"""
Benchmark de computação paralela
"""
function benchmark_parallel_computing(n = 10000000)
    using Distributed
    
    # Função para calcular π usando Monte Carlo
    function monte_carlo_pi_serial(n)
        count = 0
        for i = 1:n
            x, y = rand(), rand()
            if x^2 + y^2 <= 1
                count += 1
            end
        end
        return 4 * count / n
    end
    
    function monte_carlo_pi_parallel(n)
        count = @distributed (+) for i = 1:n
            x, y = rand(), rand()
            x^2 + y^2 <= 1 ? 1 : 0
        end
        return 4 * count / n
    end
    
    println("Benchmarking Monte Carlo π calculation with $n samples")
    
    # Serial
    serial_result = @benchmark monte_carlo_pi_serial($n)
    serial_time = median(serial_result.times) / 1e9
    
    # Parallel
    parallel_result = @benchmark monte_carlo_pi_parallel($n)
    parallel_time = median(parallel_result.times) / 1e9
    
    speedup = serial_time / parallel_time
    
    println("Serial time: $serial_time s")
    println("Parallel time: $parallel_time s")
    println("Speedup: $(speedup)x")
    
    return (
        serial_time = serial_time,
        parallel_time = parallel_time,
        speedup = speedup
    )
end

"""
Benchmark de GPU computing
"""
function benchmark_gpu_computing(n = 1000)
    using CUDA
    
    if !CUDA.functional()
        println("CUDA não disponível")
        return nothing
    end
    
    println("Benchmarking GPU vs CPU matrix operations for size $n x $n")
    
    # Matrizes CPU
    A_cpu = randn(Float32, n, n)
    B_cpu = randn(Float32, n, n)
    
    # Matrizes GPU
    A_gpu = CuArray(A_cpu)
    B_gpu = CuArray(B_cpu)
    
    # Benchmark CPU
    cpu_result = @benchmark $A_cpu * $B_cpu
    cpu_time = median(cpu_result.times) / 1e9
    
    # Benchmark GPU
    gpu_result = @benchmark CUDA.@sync $A_gpu * $B_gpu
    gpu_time = median(gpu_result.times) / 1e9
    
    speedup = cpu_time / gpu_time
    
    println("CPU time: $cpu_time s")
    println("GPU time: $gpu_time s")
    println("GPU speedup: $(speedup)x")
    
    return (
        cpu_time = cpu_time,
        gpu_time = gpu_time,
        speedup = speedup
    )
end

end # module
```

### 🔬 Simulações Científicas

**Simulação de Dinâmica Molecular**
```julia
module MolecularDynamics

using LinearAlgebra, Random, Plots

"""
Simulação de dinâmica molecular usando potencial de Lennard-Jones
"""
mutable struct MDSystem
    positions::Matrix{Float64}  # 3 x N matrix
    velocities::Matrix{Float64}  # 3 x N matrix
    forces::Matrix{Float64}     # 3 x N matrix
    masses::Vector{Float64}     # N vector
    box_size::Float64
    n_particles::Int
    
    function MDSystem(n_particles, box_size, temperature=1.0)
        # Inicializar posições em uma grade cúbica
        positions = initialize_positions(n_particles, box_size)
        
        # Inicializar velocidades com distribuição de Maxwell-Boltzmann
        velocities = initialize_velocities(n_particles, temperature)
        
        # Forças iniciais (serão calculadas)
        forces = zeros(3, n_particles)
        
        # Massas unitárias
        masses = ones(n_particles)
        
        new(positions, velocities, forces, masses, box_size, n_particles)
    end
end

function initialize_positions(n_particles, box_size)
    # Arranjo cúbico simples
    n_per_side = ceil(Int, n_particles^(1/3))
    spacing = box_size / n_per_side
    
    positions = zeros(3, n_particles)
    particle_idx = 1
    
    for i = 1:n_per_side, j = 1:n_per_side, k = 1:n_per_side
        if particle_idx > n_particles
            break
        end
        
        positions[:, particle_idx] = [
            (i - 0.5) * spacing,
            (j - 0.5) * spacing,
            (k - 0.5) * spacing
        ]
        particle_idx += 1
    end
    
    return positions
end

function initialize_velocities(n_particles, temperature)
    # Distribuição de Maxwell-Boltzmann
    velocities = randn(3, n_particles) * sqrt(temperature)
    
    # Remover momento linear total
    total_momentum = sum(velocities, dims=2)
    velocities .- total_momentum / n_particles
    
    return velocities
end

"""
Calcular forças usando potencial de Lennard-Jones
"""
function calculate_forces!(system::MDSystem; σ=1.0, ε=1.0, cutoff=2.5)
    fill!(system.forces, 0.0)
    
    n = system.n_particles
    box = system.box_size
    
    for i = 1:n-1
        for j = i+1:n
            # Vetor distância com condições periódicas de contorno
            dr = system.positions[:, j] - system.positions[:, i]
            
            # Aplicar condições periódicas
            for dim = 1:3
                if dr[dim] > box/2
                    dr[dim] -= box
                elseif dr[dim] < -box/2
                    dr[dim] += box
                end
            end
            
            r = norm(dr)
            
            if r < cutoff
                # Potencial de Lennard-Jones: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
                # Força: F = -dV/dr
                
                σ_over_r = σ / r
                σ_over_r6 = σ_over_r^6
                σ_over_r12 = σ_over_r6^2
                
                # Magnitude da força
                force_magnitude = 24 * ε * (2 * σ_over_r12 - σ_over_r6) / r^2
                
                # Vetor força
                force_vector = force_magnitude * dr / r
                
                # Aplicar terceira lei de Newton
                system.forces[:, i] += force_vector
                system.forces[:, j] -= force_vector
            end
        end
    end
end

"""
Integração usando algoritmo de Verlet
"""
function verlet_step!(system::MDSystem, dt::Float64)
    # Atualizar posições
    system.positions += system.velocities * dt + 0.5 * system.forces * dt^2
    
    # Aplicar condições periódicas de contorno
    for i = 1:system.n_particles
        for dim = 1:3
            if system.positions[dim, i] > system.box_size
                system.positions[dim, i] -= system.box_size
            elseif system.positions[dim, i] < 0
                system.positions[dim, i] += system.box_size
            end
        end
    end
    
    # Salvar forças antigas
    old_forces = copy(system.forces)
    
    # Calcular novas forças
    calculate_forces!(system)
    
    # Atualizar velocidades
    system.velocities += 0.5 * (old_forces + system.forces) * dt
end

"""
Executar simulação de dinâmica molecular
"""
function run_simulation(system::MDSystem, n_steps::Int, dt::Float64;
                       output_frequency::Int = 100)
    
    # Arrays para armazenar propriedades
    times = Float64[]
    kinetic_energies = Float64[]
    potential_energies = Float64[]
    temperatures = Float64[]
    
    println("Iniciando simulação MD com $(system.n_particles) partículas")
    println("Passos: $n_steps, dt: $dt")
    
    for step = 1:n_steps
        # Passo de integração
        verlet_step!(system, dt)
        
        # Calcular propriedades
        if step % output_frequency == 0
            t = step * dt
            ke = kinetic_energy(system)
            pe = potential_energy(system)
            temp = temperature(system)
            
            push!(times, t)
            push!(kinetic_energies, ke)
            push!(potential_energies, pe)
            push!(temperatures, temp)
            
            println("Step $step: T=$temp, KE=$ke, PE=$pe, Total=$(ke+pe)")
        end
    end
    
    return (
        times = times,
        kinetic_energy = kinetic_energies,
        potential_energy = potential_energies,
        temperature = temperatures,
        final_positions = copy(system.positions),
        final_velocities = copy(system.velocities)
    )
end

function kinetic_energy(system::MDSystem)
    ke = 0.0
    for i = 1:system.n_particles
        v_squared = sum(system.velocities[:, i].^2)
        ke += 0.5 * system.masses[i] * v_squared
    end
    return ke
end

function potential_energy(system::MDSystem; σ=1.0, ε=1.0, cutoff=2.5)
    pe = 0.0
    n = system.n_particles
    box = system.box_size
    
    for i = 1:n-1
        for j = i+1:n
            dr = system.positions[:, j] - system.positions[:, i]
            
            # Condições periódicas
            for dim = 1:3
                if dr[dim] > box/2
                    dr[dim] -= box
                elseif dr[dim] < -box/2
                    dr[dim] += box
                end
            end
            
            r = norm(dr)
            
            if r < cutoff
                σ_over_r = σ / r
                σ_over_r6 = σ_over_r^6
                σ_over_r12 = σ_over_r6^2
                
                pe += 4 * ε * (σ_over_r12 - σ_over_r6)
            end
        end
    end
    
    return pe
end

function temperature(system::MDSystem)
    ke = kinetic_energy(system)
    # T = 2*KE / (3*N*k_B), assumindo k_B = 1
    return 2 * ke / (3 * system.n_particles)
end

end # module
```

### 🎯 Competências Demonstradas

#### Computação Científica
- ✅ **Álgebra Linear**: Decomposições, solvers iterativos, problemas de autovalores
- ✅ **Análise Numérica**: Integração, diferenciação, interpolação, FFT
- ✅ **Equações Diferenciais**: EDOs, EDPs, sistemas stiff, métodos adaptativos
- ✅ **Otimização**: Linear, não-linear, global, metaheurísticas

#### High Performance Computing
- ✅ **GPU Computing**: CUDA.jl para aceleração massiva
- ✅ **Computação Paralela**: Multi-threading, computação distribuída
- ✅ **Otimização de Performance**: Benchmarking, profiling, otimização de código
- ✅ **Algoritmos Paralelos**: Implementações escaláveis

#### Simulações Científicas
- ✅ **Dinâmica Molecular**: Simulações de sistemas de partículas
- ✅ **Monte Carlo**: Métodos estocásticos para integração e simulação
- ✅ **Física Computacional**: Mecânica quântica, dinâmica de fluidos
- ✅ **Modelagem Matemática**: Sistemas dinâmicos, redes complexas

---

## 🇺🇸 English

### 🧮 Overview

Comprehensive **scientific computing** platform developed in Julia for high-performance numerical analysis:

- 🔢 **Linear Algebra**: Optimized matrix operations and decompositions
- 📊 **Numerical Analysis**: Advanced numerical methods and double precision
- 🌊 **Differential Equations**: Solvers for ODEs, PDEs and dynamical systems
- 🎯 **Optimization**: Linear and nonlinear optimization algorithms
- 🔬 **Simulations**: Monte Carlo, molecular dynamics, computational physics
- ⚡ **Parallel Computing**: GPU computing and distributed processing

### 🎯 Platform Objectives

- **Accelerate computations** scientific with performance close to C
- **Implement algorithms** state-of-the-art numerical
- **Facilitate simulations** complex in physics and engineering
- **Optimize problems** large-scale with advanced methods
- **Democratize HPC** with friendly interface and clear documentation

### 🧮 Implemented Algorithms

#### 1. 🔢 Advanced Linear Algebra
- Optimized matrix decompositions (SVD, QR, LU)
- Iterative solvers for large sparse systems
- Eigenvalue problems for large matrices
- GPU-accelerated linear algebra operations

#### 2. 🌊 Advanced Differential Equations
- Adaptive solvers for stiff ODE systems
- Finite difference methods for PDEs
- Stochastic differential equations
- Boundary value problems

#### 3. 🎯 Global Optimization
- Genetic algorithms for global optimization
- Particle swarm optimization (PSO)
- Simulated annealing
- Multi-objective optimization

### 🎯 Skills Demonstrated

#### Scientific Computing
- ✅ **Linear Algebra**: Decompositions, iterative solvers, eigenvalue problems
- ✅ **Numerical Analysis**: Integration, differentiation, interpolation, FFT
- ✅ **Differential Equations**: ODEs, PDEs, stiff systems, adaptive methods
- ✅ **Optimization**: Linear, nonlinear, global, metaheuristics

#### High Performance Computing
- ✅ **GPU Computing**: CUDA.jl for massive acceleration
- ✅ **Parallel Computing**: Multi-threading, distributed computing
- ✅ **Performance Optimization**: Benchmarking, profiling, code optimization
- ✅ **Parallel Algorithms**: Scalable implementations

#### Scientific Simulations
- ✅ **Molecular Dynamics**: Particle system simulations
- ✅ **Monte Carlo**: Stochastic methods for integration and simulation
- ✅ **Computational Physics**: Quantum mechanics, fluid dynamics
- ✅ **Mathematical Modeling**: Dynamical systems, complex networks

---

## 📄 Licença | License

MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes | see [LICENSE](LICENSE) file for details

## 📞 Contato | Contact

**GitHub**: [@galafis](https://github.com/galafis)  
**LinkedIn**: [Gabriel Demetrios Lafis](https://linkedin.com/in/galafis)  
**Email**: gabriel.lafis@example.com

---

<div align="center">

**Desenvolvido com ❤️ para Computação Científica | Developed with ❤️ for Scientific Computing**


</div>

