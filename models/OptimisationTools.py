from nubo.acquisition import UpperConfidenceBound
from nubo.models import GaussianProcess, fit_gp
from nubo.optimisation import single
from gpytorch.likelihoods import GaussianLikelihood
import torch
from nubo.test_functions import Hartmann6D
from nubo.utils import gen_inputs

# PRACTISE SO IK HOW THIS WORKS LOL
# test function
# NOTE SO U DONT FORGET, MAKE SURER U MAKE AUTO RECEPTANCE PLOTS TONIGHT ASWELL!!!! FOR PARAMETER SWEEPING

func = Hartmann6D(minimise=False)
dims = 6

# specify bounds
bounds = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1.]])
# Bayesian optimisation loop
iters = 40

# training data
x_train = gen_inputs(num_points=dims*5,
                     num_dims=dims,
                     bounds=bounds)
y_train = func(x_train)
for iter in range(iters):
    
    # specify Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)
    
    # fit Gaussian process
    fit_gp(x_train, y_train, gp=gp, likelihood=likelihood, lr=0.1, steps=200)

    # specify acquisition function
    acq = UpperConfidenceBound(gp=gp, beta=1.96**2)

    # optimise acquisition function
    x_new, _ = single(func=acq, method="L-BFGS-B", bounds=bounds, num_starts=5)

    # evaluate new point
    y_new = func(x_new)
    
    # add to data
    x_train = torch.vstack((x_train, x_new))
    y_train = torch.hstack((y_train, y_new))

    # print new best
    if y_new > torch.max(y_train[:-1]):
        print(f"New best at evaluation {len(y_train)}: \t Inputs: {x_new.numpy().reshape(dims).round(4)}, \t Outputs: {-y_new.numpy().round(4)}")