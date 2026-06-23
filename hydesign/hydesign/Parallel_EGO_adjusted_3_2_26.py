# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:44:06 2023
Updated: Fixed numpy scalar error in perturbe_around_point

@author: mikf & jumu & thijs
"""
import os
import time
from multiprocessing import Pool
from sys import version_info

import numpy as np
import pandas as pd
import smt
from numpy import newaxis as na
from openmdao.core.driver import Driver

from scipy import optimize
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from smt.applications.ego import Evaluator
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSurrogateModel,
)

# SMT imports
from smt.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)
from smt.sampling_methods import LHS, FullFactorial, Random
from smt.surrogate_models import GEKPLS, KPLS, KPLSK, KRG

# HyDesign imports
from hydesign.examples import examples_filepath

smt_version = smt.__version__.split(".")
smt_major, smt_minor = smt_version[:2]

def LCB(sm, point):
    """Lower confidence bound optimization"""
    pred = sm.predict_values(point)
    var = sm.predict_variances(point)
    res = pred - 3.0 * np.sqrt(var)
    return res

def EI(sm, point, fmin=1e3):
    """Negative Expected improvement"""
    pred = sm.predict_values(point)
    sig = np.sqrt(sm.predict_variances(point))

    args0 = (fmin - pred) / sig
    args1 = (fmin - pred) * norm.cdf(args0)
    args2 = sig * norm.pdf(args0)
    ei = args1 + args2
    return -ei

def KStd(sm, point):
    """Prediction Error"""
    res = np.sqrt(sm.predict_variances(point))
    return res

def KB(sm, point):
    """Mean GP process"""
    res = sm.predict_values(point)
    return res

def get_sm(xdoe, ydoe, theta_bounds=[1e-06, 2e1], n_comp=4):
    """
    Train the surrogate model.
    Switched to KRG (Standard Kriging) to avoid KPLSK inheritance issues in SMT.
    """
    sm = KRG(
        corr="squar_exp",
        poly="linear",
        theta0=[1e-2],
        theta_bounds=theta_bounds,
        print_global=False,
    )
    sm.set_training_values(xdoe, ydoe)
    sm.train()
    return sm

def eval_sm(sm, mixint, scaler=None, seed=0, npred=1e3, fmin=1e10):
    sampling = get_sampling(mixint, seed=int(seed), criterion="c")
    xpred = sampling(int(npred))
    xpred = np.array(mixint.design_space.decode_values(xpred))

    if scaler is not None:
        xpred = scaler.transform(xpred)

    ypred_LB = EI(sm=sm, point=xpred, fmin=fmin)
    return xpred, ypred_LB

def opt_sm_EI(sm, mixint, x0, fmin=1e10, n_seed=0):
    ndims = mixint.get_unfolded_dimension()
    func = lambda x: EI(sm, x[np.newaxis, :], fmin=fmin)

    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": [(0, 1)] * ndims,
        "options": {"maxiter": 20, "eps": 1e-3, "disp": False},
    }

    res = optimize.basinhopping(
        func,
        x0=x0,
        niter=100,
        stepsize=10,
        minimizer_kwargs=minimizer_kwargs,
        seed=n_seed,
        target_accept_rate=0.5,
        stepwise_factor=0.9,
    )
    return res.x.reshape([1, -1])

def opt_sm(sm, mixint, x0, fmin=1e10):
    ndims = mixint.get_unfolded_dimension()
    res = optimize.minimize(
        fun=lambda x: KB(sm, x.reshape([1, ndims])),
        jac=lambda x: np.stack(
            [sm.predict_derivatives(x.reshape([1, ndims]), kx=i) for i in range(ndims)]
        ).reshape([1, ndims]),
        x0=x0.reshape([1, ndims]),
        method="SLSQP",
        bounds=[(0, 1)] * ndims,
        options={"maxiter": 20, "eps": 1e-4, "disp": False},
    )
    return res.x.reshape([1, -1])

def get_candiate_points(x, y, quantile=0.25, n_clusters=32):
    yq = np.quantile(y, quantile)
    ind_up = np.where(y < yq)[0]
    xup = x[ind_up]
    yup = y[ind_up]
    
    n_points = xup.shape[0]
    if n_points < n_clusters:
        n_clusters = max(1, n_points)
        
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=10,
    ).fit(xup)
    clust_id = kmeans.predict(xup)
    xbest_per_clst = np.vstack(
        [
            xup[np.where(yup == np.min(yup[np.where(clust_id == i)[0]]))[0], :]
            for i in range(n_clusters)
        ]
    )
    return xbest_per_clst

def extreme_around_point(x):
    ndims = x.shape[1]
    xcand = np.tile(x.T, ndims * 2).T
    for i in range(ndims):
        xcand[i, i] = 0.0
    for i in range(ndims):
        xcand[i + ndims, i] = 1.0
    return xcand

def perturbe_around_point(x, step=0.1):
    ndims = x.shape[1]
    xcand = np.tile(x.T, ndims * 2).T
    for i in range(ndims):
        xcand[i, i] += step
    for i in range(ndims):
        xcand[i + ndims, i] -= step

    xcand = np.maximum(xcand, 0)
    xcand = np.minimum(xcand, 1.0)
    return xcand

def get_design_vars(variables):
    return [
        var_ for var_ in variables.keys() if variables[var_]["var_type"] == "design"
    ], [var_ for var_ in variables.keys() if variables[var_]["var_type"] == "fixed"]

def get_limits(variables, design_var=[]):
    if len(design_var) == 0:
        design_var, fixed_var = get_design_vars(variables)
    return np.array([variables[var_]["limits"] for var_ in design_var])

def drop_duplicates(x, y, decimals=3):
    x_rounded = np.around(x, decimals=decimals)
    _, indices = np.unique(x_rounded, axis=0, return_index=True)
    x_unique = x[indices, :]
    y_unique = y[indices, :]
    return x_unique, y_unique

def concat_to_existing(x, y, xnew, ynew):
    x_concat, y_concat = drop_duplicates(np.vstack([x, xnew]), np.vstack([y, ynew]))
    return x_concat, y_concat

def surrogate_optimization(inputs):
    x, kwargs = inputs
    mixint = get_mixint_context(kwargs["variables"], kwargs["n_seed"])
    return opt_sm(kwargs["sm"], mixint, x, fmin=kwargs["yopt"][0, 0])

def surrogate_evaluation(inputs):
    seed, kwargs = inputs
    mixint = get_mixint_context(kwargs["variables"], kwargs["n_seed"])
    return eval_sm(
        kwargs["sm"],
        mixint,
        scaler=kwargs["scaler"],
        seed=seed,
        npred=kwargs["npred"],
        fmin=kwargs["yopt"][0, 0],
    )

def get_xlimits(variables, design_var=[]):
    if len(design_var) == 0:
        design_var, fixed_var = get_design_vars(variables)
    return np.array([variables[var_]["limits"] for var_ in design_var])

def get_xtypes(variables, design_var=[]):
    if len(design_var) == 0:
        design_var, fixed_var = get_design_vars(variables)
    return [variables[var_]["types"] for var_ in design_var]

def cast_to_mixint(x, variables):
    design_var, fixed_var = get_design_vars(variables)
    types_ = get_xtypes(variables)
    for i, ty in enumerate(types_):
        if ty == "int":
            x[:, i] = np.round(x[:, i])
        elif ty == "resolution":
            res = variables[design_var[i]]["resolution"]
            x[:, i] = np.round(x[:, i] / res, decimals=0) * res
    return x

def get_mixint_context(variables, seed=None, criterion="maximin"):
    design_var, fixed_var = get_design_vars(variables)
    list_vars_doe = []
    for var_ in design_var:
        if variables[var_]["types"] == "int":
            list_vars_doe += [IntegerVariable(*variables[var_]["limits"])]
        elif variables[var_]["types"] == "float":
            list_vars_doe += [FloatVariable(*variables[var_]["limits"])]
        else:
            dtype = type(variables[var_]["resolution"])
            val_list = list(
                np.arange(
                    variables[var_]["limits"][0],
                    variables[var_]["limits"][1] + variables[var_]["resolution"],
                    variables[var_]["resolution"],
                    dtype=dtype,
                )
            )
            list_vars_doe += [OrdinalVariable(val_list)]

    is_old_smt = (int(smt_major) == 2 and int(smt_minor) < 4)
    if is_old_smt:
        mixint = MixedIntegerContext(DesignSpace(list_vars_doe, seed=seed))
    else:
        ds = DesignSpace(list_vars_doe, random_state=seed)
        ds.sampler = LHS(
            xlimits=ds.get_unfolded_num_bounds(),
            random_state=int(seed),
            criterion=criterion,
        )
        mixint = MixedIntegerContext(ds)
    return mixint

def get_sampling(mixint, seed, criterion="maximin"):
    is_old_smt = (int(smt_major) == 2 and int(smt_minor) < 1)
    if is_old_smt:
        sampling = mixint.build_sampling_method(
            LHS, criterion=criterion, random_state=int(seed)
        )
    else:
        mixint._design_space.sampler = LHS(
            xlimits=mixint.get_unfolded_xlimits(),
            criterion=criterion,
            random_state=int(seed),
        )
        sampling = mixint.build_sampling_method()
    return sampling

def expand_x_for_model_eval(x, kwargs):
    list_vars = kwargs["list_vars"]
    variables = kwargs["variables"]
    design_vars = kwargs["design_vars"]
    fixed_vars = kwargs["fixed_vars"]
    x_eval = np.zeros([x.shape[0], len(list_vars)])
    for ii, var in enumerate(list_vars):
        if var in design_vars:
            x_eval[:, ii] = x[:, design_vars.index(var)]
        elif var in fixed_vars:
            x_eval[:, ii] = variables[var]["value"]
    return x_eval

def model_evaluation(inputs):
    x, kwargs = inputs
    hpp_m = kwargs["hpp_model"](**kwargs, verbose=False)
    x = kwargs["scaler"].inverse_transform(x)
    x_eval = expand_x_for_model_eval(x, kwargs)
    try:
        return np.array(
            kwargs["opt_sign"] * hpp_m.evaluate(*x_eval[0, :])[kwargs["op_var_index"]]
        )
    except:
        print("There was an error with this case (or potentially memory error): ")
        print("x=[" + ", ".join(map(str, x_eval[0, :])) + "]")
        return np.array([1e10])

class ParallelEvaluator(Evaluator):
    def __init__(self, n_procs=31):
        self.n_procs = n_procs

    def run_ydoe(self, fun, x, **kwargs):
        n_procs = self.n_procs
        with Pool(n_procs) as p:
            return np.array(
                p.map(fun, [(x[[i], :], kwargs) for i in range(x.shape[0])])
            ).reshape(-1, 1)

    def run_both(self, fun, i, **kwargs):
        n_procs = self.n_procs
        with Pool(n_procs) as p:
            return p.map(
                fun,
                [
                    ((n + i * n_procs) * 100 + kwargs["n_seed"], kwargs)
                    for n in np.arange(n_procs)
                ],
            )

    def run_xopt_iter(self, fun, x, **kwargs):
        n_procs = self.n_procs
        with Pool(n_procs) as p:
            return np.vstack(
                p.map(fun, [(x[[ii], :], kwargs) for ii in range(x.shape[0])])
            )

def check_types(kwargs):
    for x in ["num_batteries", "n_procs", "n_doe", "n_clusters", "n_seed", "max_iter"]:
        kwargs[x] = int(kwargs[x])
    if kwargs["final_design_fn"] == None:
        kwargs["final_design_fn"] = (
            f'{kwargs["work_dir"]}design_hpp_{kwargs["name"]}_{kwargs["opt_var"]}.csv'
        )
    for x in ["opt_var", "final_design_fn"]:
        kwargs[x] = str(kwargs[x])
    return kwargs

class EfficientGlobalOptimizationDriver(Driver):
    def __init__(self, **kwargs):
        os.environ["OPENMDAO_USE_MPI"] = "0"
        kwargs = check_types(kwargs)
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _declare_options(self):
        for k, v in self.kwargs.items():
            self.options.declare(k, v)

    def run(self):
        kwargs = self.kwargs
        recorder = {"time": [], "yopt": []}
        start_total = time.time()

        variables = kwargs["variables"]
        design_vars, fixed_vars = get_design_vars(variables)
        xlimits = get_xlimits(variables, design_vars)
        xtypes = get_xtypes(variables, design_vars)

        scaler = MinMaxScaler()
        scaler.fit(xlimits.T)

        mixint = get_mixint_context(kwargs["variables"], kwargs["n_seed"])
        sampling = get_sampling(mixint, seed=kwargs["n_seed"], criterion="maximin")
        xdoe = sampling(kwargs["n_doe"])
        xdoe = np.array(mixint.design_space.decode_values(xdoe))

        self.xdoe = xdoe
        xdoe = scaler.transform(xdoe)
        
        name = kwargs["name"]
        print("\n\n\nSizing a HPP plant at {}:".format(name))
        
        opt_var = kwargs["opt_var"]
        opt_sign = -1 
        if "LCOE" in opt_var:
            opt_sign = 1

        kwargs["opt_sign"] = opt_sign
        kwargs["scaler"] = scaler
        kwargs["xtypes"] = xtypes
        kwargs["xlimits"] = xlimits

        hpp_m = kwargs["hpp_model"](**kwargs)
        kwargs["input_ts_fn"] = hpp_m.input_ts_fn
        kwargs["altitude"] = hpp_m.altitude
        kwargs["price_fn"] = None

        print("\n\n")

        list_vars = hpp_m.list_vars
        list_out_vars = hpp_m.list_out_vars
        op_var_index = list_out_vars.index(opt_var)
        kwargs.update({"op_var_index": op_var_index})

        kwargs["list_vars"] = list_vars
        kwargs["design_vars"] = design_vars
        kwargs["fixed_vars"] = fixed_vars

        start = time.time()
        n_procs = kwargs["n_procs"]
        PE = ParallelEvaluator(n_procs=n_procs)
        ydoe = PE.run_ydoe(fun=model_evaluation, x=xdoe, **kwargs)

        lapse = np.round((time.time() - start) / 60, 2)
        print(f"Initial {xdoe.shape[0]} simulations took {lapse} minutes")

        itr = 0
        error = 1e10
        conv_iter = 0
        xopt = xdoe[[np.argmin(ydoe)], :]
        yopt = ydoe[[np.argmin(ydoe)], :]
        kwargs["yopt"] = yopt
        yold = np.copy(yopt)
        
        print(f"  Current solution {opt_sign}*{opt_var} = {float(np.squeeze(yopt)):.3E}".replace("1*", ""))
        print(f"  Current No. model evals: {xdoe.shape[0]}\n")
        
        recorder["time"].append(time.time())
        recorder["yopt"].append(float(np.squeeze(yopt)))
        
        sm_args = {"n_comp": min(len(design_vars), 4)}
        sm_args.update({k: v for k, v in kwargs.items() if k in ["theta_bounds", "n_comp"]})
        
        while itr < kwargs["max_iter"]:
            start_iter = time.time()
            np.random.seed(kwargs["n_seed"])
            sm = get_sm(xdoe, ydoe, **sm_args)
            kwargs["sm"] = sm

            start = time.time()
            both = PE.run_both(surrogate_evaluation, itr, **kwargs)
            xpred = np.vstack([both[ii][0] for ii in range(len(both))])
            ypred_LB = np.vstack([both[ii][1] for ii in range(len(both))])

            n_clusters = kwargs["n_clusters"]
            xnew = get_candiate_points(xpred, ypred_LB, n_clusters=n_clusters, quantile=1e-2)
            lapse = np.round((time.time() - start) / 60, 2)
            print(f"Update sm and extract candidate points took {lapse} minutes")

            if np.abs(error) < kwargs["tol"]:
                np.random.seed(kwargs["n_seed"] * 100 + itr)
                # FIX: Generating a scalar float instead of a numpy array
                step = np.random.uniform(low=0.05, high=0.25) 
                xopt_iter = perturbe_around_point(xopt, step=step)
            else:
                xopt_iter = extreme_around_point(xopt)

            xopt_iter = scaler.inverse_transform(xopt_iter)
            xopt_iter = cast_to_mixint(xopt_iter, kwargs["variables"])
            xopt_iter = scaler.transform(xopt_iter)
            xopt_iter, _ = drop_duplicates(xopt_iter, np.zeros_like(xopt_iter))
            xopt_iter, _ = concat_to_existing(xnew, np.zeros_like(xnew), xopt_iter, np.zeros_like(xopt_iter))

            start = time.time()
            yopt_iter = PE.run_ydoe(fun=model_evaluation, x=xopt_iter, **kwargs)
            lapse = np.round((time.time() - start) / 60, 2)
            print(f"Check-optimal candidates: new {xopt_iter.shape[0]} simulations took {lapse} minutes")

            xdoe_upd, ydoe_upd = concat_to_existing(xdoe, ydoe, xopt_iter, yopt_iter)
            xdoe_upd, ydoe_upd = drop_duplicates(xdoe_upd, ydoe_upd)

            xopt = xdoe_upd[[np.argmin(ydoe_upd)], :]
            yopt = ydoe_upd[[np.argmin(ydoe_upd)], :]

            recorder["time"].append(time.time())
            recorder["yopt"].append(float(np.squeeze(yopt)))

            error = opt_sign * float(1 - (np.squeeze(yold) / np.squeeze(yopt)))

            xdoe = np.copy(xdoe_upd)
            ydoe = np.copy(ydoe_upd)
            yold = np.copy(yopt)
            itr = itr + 1
            lapse = np.round((time.time() - start_iter) / 60, 2)

            print(f"  Current solution {opt_sign}*{opt_var} = {float(np.squeeze(yopt)):.3E}".replace("1*", ""))
            print(f"  Current No. model evals: {xdoe.shape[0]}")
            print(f"  rel_yopt_change = {error:.2E}")
            print(f"Iteration {itr} took {lapse} minutes\n")

            if np.abs(error) < kwargs["tol"]:
                conv_iter += 1
                if conv_iter >= kwargs["min_conv_iter"]:
                    print("Surrogate based optimization is converged.")
                    break
            else:
                conv_iter = 0

        xopt = scaler.inverse_transform(xopt)
        xopt = expand_x_for_model_eval(xopt, kwargs)

        outs = hpp_m.evaluate(*xopt[0, :])
        yopt = np.array(opt_sign * outs[[op_var_index]])[:, na]
        hpp_m.print_design(xopt[0, :], outs)

        recorder["time"].append(time.time())
        recorder["yopt"].append(float(np.squeeze(yopt)))

        n_model_evals = xdoe.shape[0]
        lapse = np.round((time.time() - start_total) / 60, 2)
        print(f"Optimization with {itr} iterations and {n_model_evals} model evaluations took {lapse} minutes\n")

        design_df = pd.DataFrame(columns=list_vars, index=[name])
        for var_ in ["name", "longitude", "latitude", "altitude"]:
            design_df[var_] = kwargs[var_]
        for iv, var in enumerate(list_vars):
            design_df[var] = xopt[0, iv]
        for iv, var in enumerate(list_out_vars):
            design_df[var] = outs[iv]

        design_df["design obj"] = opt_var
        design_df["opt time [min]"] = lapse
        design_df["n_model_evals"] = n_model_evals

        design_df.T.to_csv(kwargs["final_design_fn"])
        self.result = design_df
        self.hpp_m = hpp_m
        self.recorder = recorder