# A high performance implementation of Shapley effects in Julia for Scientific machine learning

Devang Sehgal<sup>1</sup> &dagger; and Anurag Vaidya<sup>1</sup> &dagger;

[1] Department of Health Science and Technology, MIT, Cambridge  
[&dagger;] Equal contribution 

# Introduction

This library implements an optimized serial version and a parallel version of Shapley effects introduced by [1]. Unlike other state of the art sensitivity analysis methods, like Sobol indices, the Shapley effects algorithm can take into account correlations between different input features. Our algorithm requires the user to define a numerical or probabilistic function, the marginal distirbution of each of the input features, a Copula to denote how the different features interact, and give the number of boot-strapped to be used to calculate the Shapley effects for each feature. The algorithm returns Shapley effects such that they add upto 1, which means that each effect can be interpreted as what percentage of variance is accounted for by that specific feature. This implementation of Shapley effects is inspired from the R code and pseudo-code provided by Song et al. [1] and Shapley effects implementation in Python [2]. Our implementation havily uses the Copulas.jl [3] and Distributions.jl [4]

# Salient features of our implementation
Some of the salient features of our algorithm implementationa are: 
- Serial and parallel optimized code provided in separate files 
- Compared to the python counterpart of the algorithm written using numpy, the julia version is 8x faster in computation speed
- Sample generation and bootstrapping is separated unlike in the original implementation of [1]. This allows modular control over each part of the algorithm 
- Our implementation can be applied to analytical functions and differential equations, with examples shown in the attached report analyzing the algorithm in detial.
- The algorithm has factorial complexity with respect to the number of features in the function, making it intractable to work with functions with large input features. Thus, we implement another version of the method (random permutation) which samples a user defined number of permutations, making the algorithm tractable for functions with large number of inputs.

# Repo sturcture
- `shapley` directory contains the serial and parallel versions of the algorithm (`shapley_serial.jl` and `shapley_parallel.jl` respectively)
- `examples` directory contains some analytical functions (`run_linear.jl`, `run_ishi.jl`) for both batch and non-batch version. 
- `examples` directory also has a differential equation system (`run_prey_predator.jl`) 
- `examples` also has a file called `rand_perm.jl` that exemplifies how to randomly sample the number of feature permutations 

# Example

First define the function you want to test. Here we work with the Ishigami fucntion, but this function can be also be a differential equation. 

```
using Copulas, Distributions
include("path/to/shapley_serial.jl")

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end
```

Now, define the hyper-parameters:
```
n_perms = -1; # -1 indicates that we want to consider all permutations. One can also use n_perms > 0
n_var = 1000;
n_outer = 100;
n_inner = 3
n_boot = 60_000;
dim = 3;
```

Let's define unifrom distirbutions for the marginal distributions of the input features. Let's assume there is no correlation between the features
```
margins = (Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi));
dependency_matrix = 1* Matrix(I, dim, dim)

C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C,margins);
```

Now, we define the Shapley method with the hyper-parameters

```
method = Shapley(dim=dim, n_perms=n_perms, n_var = n_var, n_outer = n_outer, n_inner = n_inner, n_boot=n_boot);
```

Let's get the results on the non-batched version of the Ishigami function.
```
result_non_batch = gsa(ishi,method,input_distribution,batch=false)
shapley_indices = result_non_batch.Shapley_indices
```

To report the final Shapley effect for each feature, we find the median value over all boot-strapped samples.
```
for i in range(1, dim)
    println("Median Shapley effect for feature $i = ", median(shapley_indices[i, :]))
end
```

We can also define the confidence interval for each Shapley effect.
```
function t_test(x; conf_level=0.95)
    alpha = (1 - conf_level)
    tstar = quantile(TDist(length(x)-1), 1 - alpha/2)
    SE = std(x)/sqrt(length(x))

    lo = mean(x) + (-1 * tstar * SE)
    hi = mean(x) + (1 * tstar * SE)

    tstar * SE
end

for i in range(1, dim)
    println("95% CI for Shapley effect of feature $i = ", t_test(shapley_indices[i, :]))
end

```

# License 
The code is made available under the MIT License 2023 &copy;

# References
[1] Song, Eunhye, Barry L. Nelson, and Jeremy Staum. "Shapley effects for global sensitivity analysis: Theory and computation." SIAM/ASA Journal on Uncertainty Quantification 4.1 (2016): 1060-1083.
APA	
[2] https://gitlab.com/CEMRACS17/shapley-effects
[3] https://github.com/lrnv/Copulas.jl
[4] https://github.com/JuliaStats/Distributions.jl

# Acknowledgements 
This project was originally inspired from a Google summer of code project. We would like to sincerely thank Vaibhav Dixit of the Julia team for all the guidance in this project. 
