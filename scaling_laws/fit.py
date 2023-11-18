from dataclasses import dataclass, astupleimport os
from typing import List, Dict
import operator
import itertools

from tqdm import tqdm

import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import plotly
import plotly.graph_objects as go

@dataclass
class Params:
    """
    Chinchilla scaling law parameters
    """
    a: float
    b: float
    e: float
    alpha: float
    beta: float
    

def logsumexp(a, axis=None):
    """
    Define custom logsumexp differentiable by autograd.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    a_shifted = a - a_max  # improves numerical stability
    exp_a_shifted = np.exp(a_shifted)
    sum_exp_a_shifted = np.sum(exp_a_shifted, axis=axis, keepdims=True)
    log_sum_exp = np.log(sum_exp_a_shifted) + a_max  # undo the shift

    if axis is not None:
        return np.squeeze(log_sum_exp, axis=axis)
    else:
        return log_sum_exp

def huber(residuals, delta=1e-3):
    """
    Define custom Huber loss differentiable by autograd.
    """
    abs_residuals = np.abs(residuals)
    quadratic = np.where(abs_residuals <= delta, 0.5 * np.square(abs_residuals), 0.0)
    linear = np.where(abs_residuals > delta, delta * (abs_residuals - 0.5 * delta), 0.0)
    return quadratic + linear


def loss_fn(params: Params, N: float, D: float, L: float):
    a, b, e, alpha, beta = astuple(params)
    arg1 = a - alpha * np.log(N)
    arg2 = b - beta * np.log(D)
    arg3 = e + np.zeros(arg2.shape)
    
    huber_input = logsumexp(np.stack((arg1, arg2, arg3), axis=0), axis=0) - np.log(L)
    return np.mean(huber(huber_input))

def scaling_law(params: Params, N: float, D: float):
    a, b, e, alpha, beta = astuple(params)
    return float(np.exp(e)) + float(np.exp(a))/N**alpha + float(np.exp(b))/D**beta

def fit_scaling_law(
        runs: List[Dict[str, float]]
        grid_search: bool = False
    ):
    loss_grad = grad(loss_fn, argnum=0)

    best_loss = float('inf')
    best_init = None
    best_result = None

    if grid_search:
        init_grid = dict(
            a=np.arange(0, 30, 5),
            b=np.arange(0, 30, 5),
            e=np.arange(-1, 1.5, 0.5),
            alpha=np.arange(0, 2.5, 0.5),
            beta=np.arange(0, 2.5, 0.5),
        )
    else:
        init_grid = dict(a=[10], b=[20], e=[1], alpha=[0], beta=[1]) 

    num_inits = reduce(operator.mul, [len(v) for k,v in init_grid.items()], 1)

    for x0 in tqdm(itertools.product(*[v for k,v in init_grid.items()]), total=num_inits):
        x0 = np.array(x0)
        result = minimize(
            parametric_loss, 
            x0,
            args=(runs_single_epoch['N'], runs_single_epoch['D'], runs_single_epoch['L']),
            method='L-BFGS-B',
            jac=loss_grad,
            options={'gtol': 1e-12, 'ftol': 1e-12}
        )

        if result.fun < best_loss:
            best_loss = result.fun
            best_init = x0
            best_result = result

    return Params(*best_result)
