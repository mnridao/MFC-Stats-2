# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:38:22 2025

@author: mn1215
"""

import numpy as np

import src.utils as utils
import demos.plotters as plotters

def runFilter(methodName):
    """ 
    Run the chosen filter for the toy model discussed in the report.
    Args:
        methodName: (string) name of the filter method"""
    
    filterObj = utils.setupToyProblem(methodName)
    filterObj.run()
    
    plotters.plotFilterResults(filterObj)

def runParticleFilterEnsemble(k, T, numParticlesLst, numRuns):
    """ 
    Ensemble run of the particle filter. Called to find how the marginal 
    likelihood changes. Also runs the kalman filter to compare."""
    
    kalmanFilter = utils.setupToyProblem("kalman", k, T)
    kalmanFilter.run()
    # print(f"kalman filter likelihood: {kalmanFilter.logli}")
    
    likelihoods = np.zeros((numRuns, len(numParticlesLst)))
    for i in range(numRuns):
        
        for j, numParticles in enumerate(numParticlesLst):
            
            # print(f"Run no: {i}, {numParticles=}")
            particleFilter = utils.setupToyProblem("particle", k, T)
            particleFilter.numParticles = numParticles
            particleFilter.run()
            
            likelihoods[i, j] = particleFilter.logli
    
    return likelihoods, kalmanFilter.logli

def printMeans(likelihoods):
    """ 
    """
    print(f"Means = {np.mean(likelihoods, axis=0)}")
    
def printStds(likelihoods):
    """ 
    """
    print(f"Stds = {np.std(likelihoods, axis=0)}")
#%%
if __name__ == "__main__":
    
    runFilter("kalman")
    runFilter("particle")
        
    #%%
    k = 0.04 
    T = 50
    
    numParticlesLst = [1000, 5000, 10000]
    numRuns = 10
    likelihoods, klogli = runParticleFilterEnsemble(k, T, numParticlesLst, numRuns)
    plotters.plotBoxes(likelihoods, klogli, numParticlesLst)
    
    print(np.mean(likelihoods, axis=0))
    print(np.std(likelihoods, axis=0))
        
    #%%
    numRuns = 5
    likelihoods, klogli = runParticleFilterEnsemble(k, T, numParticlesLst, numRuns)
    plotters.plotBoxes(likelihoods, klogli, numParticlesLst)
    
    print(np.mean(likelihoods, axis=0))
    print(np.std(likelihoods, axis=0))
    
    #%%
    numRuns = 20
    likelihoods, klogli = runParticleFilterEnsemble(k, T, numParticlesLst, numRuns)
    plotters.plotBoxes(likelihoods, klogli, numParticlesLst)
    
    print(np.mean(likelihoods, axis=0))
    print(np.std(likelihoods, axis=0))
    
    #%%
    import matplotlib.pyplot as plt
    
    diff = np.linalg.norm(klogli - np.mean(likelihoods, axis=0))  # Compute L2 norm
    particles = np.array(numParticlesLst)
    
    plt.figure()
    plt.plot(particles, diff, marker='o')
    plt.xlabel("Number of Particles")
    plt.ylabel("L2 Norm")
    plt.title("L2 Norm of Difference vs. Number of Particles")
    plt.grid(True)
    plt.show()