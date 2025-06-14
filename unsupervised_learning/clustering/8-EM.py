#!/usr/bin/env python3
"""
This module contains a function that performs
expectation maximization for a GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, total_log_like = expectation(X, pi, m, S)
    prev_like = total_log_like

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(0, total_log_like.round(5)))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        g, total_log_like = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i, total_log_like.round(5)))

        if abs(prev_like - total_log_like) <= tol:
            break
        prev_like = total_log_like

    if verbose and i % 10 != 0:
        print("Log Likelihood after {} iterations: {}".format(i, total_log_like.round(5)))

    return pi, m, S, g, total_log_like
