using Copulas, Distributions
include("path/to/shapley_serial.jl")
include("path/to/shapley_parallel.jl")

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[:,1]) + A*sin(X[:,2])^2+ B*X[:,3]^4 *sin(X[:,1])
end

n_perms = -1; 

n_var = 1000;
n_outer = 100;
n_inner = 3
n_boot = 60_000;

dim = 3;
margins = (Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi));
dependency_matrix = 1* Matrix(I, dim, dim)

C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C,margins);

method = Shapley(dim=dim, n_perms=n_perms, n_var = n_var, n_outer = n_outer, n_inner = n_inner, n_boot=n_boot);

res_serial = gsa(ishi,method,input_distribution,batch=false)
shapley_indices_serial = res_serial.Shapley_indices

for i in range(1, dim)
    println("Median Shapley effect for feature $i = ", median(shapley_indices_serial[i, :]))
end

res_parallel = gsa_parallel(ishi,method,input_distribution,batch=false)
shapley_indices_parallel = res_parallel.Shapley_indices

for i in range(1, dim)
    println("Median Shapley effect for feature $i = ", median(shapley_indices_parallel[i, :]))
end