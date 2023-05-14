using Copulas, Distributions
include("path/to/shapley_serial.jl")

function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2] #prey
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2] #predator
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(f, u0, tspan, p)
t = collect(range(0, stop = 10, length = 200))

f1 = let prob = prob, t = t
    function (p)
        prob1 = remake(prob; p = p)
        sol = solve(prob1, Tsit5(); saveat = t)
        return sol
    end
end

n_perms = -1; 
n_var = 1000; 
n_outer = 100; 
n_inner = 3; #
n_boot = 1000;

dim = 4;

margins = (Uniform(1, 5), Uniform(1, 5), Uniform(1, 5), Uniform(1, 5));
dependency_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C,margins);

method = Shapley(dim=dim, n_perms=n_perms, n_var = n_var, n_outer = n_outer, n_inner = n_inner, n_boot=n_boot);
res2 = gsa(f1,method,input_distribution,batch=false)

shapley_indices = res2.Shapley_indices # Holds the shapley indices
