"""
This is just SciPy's BFGS minimizer, rewritten to possibly accept
an approximation to the initial Hessian matrix as an argument.
The original function uses the identity matrix. This is the default
case, if no hessian is supplied.

Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import warnings
from numpy import (asarray, sqrt, Inf)
import numpy as np

try:
    from scipy.optimize._optimize import (_LineSearchError,
                                          _line_search_wolfe12, _check_unknown_options,
                                          vecnorm, _prepare_scalar_function,
                                          OptimizeResult, _status_message,
                                          OptimizeWarning)
except:
    from scipy.optimize.optimize import (_LineSearchError,
                                         _line_search_wolfe12, _check_unknown_options,
                                         vecnorm, _prepare_scalar_function,
                                         OptimizeResult, _status_message,
                                         OptimizeWarning)

_epsilon = sqrt(np.finfo(float).eps)


def minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                  disp=False, return_all=False, finite_diff_rel_step=None,
                  xrtol=0, initial_inv_hessian=None, f0=None, g0=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    args are the additional arguments to be supplied to fun and jac. It must be a list.
    e.g. if the other argument is an int x: args=[x].
    If it's a list [x,y,z]: [[x,y,z]].
    Both, by order: [x,[x,y,x]]

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
    initial_inv_hessian : np.ndarray, optional
        Initial estimate for the inverse Hessian
    f0 : float, optional
        Initial value of the cost function
    g0 : np.ndarray, optional
        Initial gradient vector
    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    # print("f0",f0)
    if f0 is None:
        # Calculate function at initial point
        old_fval = f(x0)
    else:
        # Function value at initial point has been provided. No costs
        old_fval = f0
        sf.nfev -= 1

    if g0 is None:
        # Calculate vector at initial point
        gfk = myfprime(x0)
    else:
        # Gradient vector at initial point has been provided. No costs
        assert len(g0) == len(x0)
        gfk = g0
        sf.ngev -= 1

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    if initial_inv_hessian is None:
        Hk = I
    else:
        Hk = initial_inv_hessian

    alphas = []

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        # print("Energy ",old_fval)
        # print(gnorm)
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                     old_fval, old_old_fval, amin=1e-100, amax=1e100)
            alphas.append(alpha_k)
            # print("Alpha",alpha_k)
            # print("Energy change:",old_old_fval-old_fval)
            # print("F, ",old_fval)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break
        sk = alpha_k * pk
        xkp1 = xk + sk

        if retall:
            allvecs.append(xkp1)
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        # Cryptic comment above is preserved for posterity. Future reader:
        # consider change to condition below proposed in gh-1261/gh-17345.
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                msg = "Divide-by-zero encountered: rhok assumed large"
                _print_success_message_or_warn(True, msg)
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])
        # print(Hk)

        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        intermediate_result.inv_hessian = Hk
        intermediate_result.gradient = gfk
        # print("G",gfk)

        if _call_callback_maybe_halt(callback, intermediate_result):
            break
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        #  See Chapter 5 in  P.E. Frandsen, K. Jonasson, H.B. Nielsen,
        #  O. Tingleff: "Unconstrained Optimization", IMM, DTU.  1999.
        #  These notes are available here:
        #  http://www2.imm.dtu.dk/documents/ftp/publlec.html
        if (alpha_k * vecnorm(pk) <= xrtol * (xrtol + vecnorm(xk))):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

    fval = old_fval
    # print("Alphas: ",alphas)

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def _call_callback_maybe_halt(callback, res):
    """Call wrapped callback; return True if minimization should stop.

    Parameters
    ----------
    callback : callable or None
        A user-provided callback wrapped with `_wrap_callback`
    res : OptimizeResult
        Information about the current iterate

    Returns
    -------
    halt : bool
        True if minimization should stop

    """
    if callback is None:
        return False
    try:
        callback(res)
        return False
    except StopIteration:
        callback.stop_iteration = True  # make `minimize` override status/msg
        return True


def _print_success_message_or_warn(warnflag, message, warntype=None):
    if not warnflag:
        print(message)
    else:
        warnings.warn(message, warntype or OptimizeWarning, stacklevel=3)
