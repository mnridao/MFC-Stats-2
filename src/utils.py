# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:04:47 2025

@author: mn1215
"""

import numpy as np

from src.filters import KalmanFilter, ParticleFilter
from src.model import Model, ToyModelConfig

def setupToyProblem(methodName, k=0.04, T=1000):
    """ 
    Sets up the Filter object for the toy problem discussed in the report.
    
    Args:
        methodName: (string) name of the filter method
    Returns:
        Filter object
    """
    
    # Select the filter method.
    if methodName == "kalman":
        method = KalmanFilter
    elif methodName == "particle":
        method = ParticleFilter
    else:
        raise Exception("Filter method not valid.")
        
    # Construct toy model for report.
    config = ToyModelConfig(k, T)    
    model = Model(config.x0, config.A, config.H, config.Q, config.R, config.rng)
    
    return method(model, config.mu0, config.V0, config.T)