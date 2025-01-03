# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:38:02 2025

@author: mn1215
"""

import matplotlib.pyplot as plt

def plotFilterResults(filterObj):
    """
    Plot the results of either the kalman filter or the particle filter. 
    Compares with the true trajectory and the observations, which are stored
    by the Model class instance.
    """
    # Filter posterior means.
    mu = filterObj.mu 
    
    # True states and observations.
    x = filterObj.model.getPastStates()
    y = filterObj.model.getPastObservations()
    
    plt.figure(figsize=(5, 5))
    plt.plot(mu[:, 0], mu[:, 1], 'k-', label=filterObj.__class__.__name__)
    plt.plot(x[:, 0], x[:, 1], 'b-', label="True trajectory")
    plt.plot(y[:, 0], y[:, 1], 'r.', alpha=0.3, label="Observations")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
def plotBoxes(likelihoods, klogli, numParticlesLst):
    """ 
    """
    data = [likelihoods[:, i] for i in range(likelihoods.shape[1])]
    plt.figure()
    plt.boxplot(data, patch_artist=True, labels=numParticlesLst, showfliers=False)
    plt.axhline(klogli, color='red', linestyle='--', label='Kalman Filter')
    plt.xlabel("Number of particles")
    plt.ylabel("marginal log-likelihood")
    plt.show()