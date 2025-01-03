# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:06:54 2025

@author: mn1215
"""

from abc import ABC, abstractmethod
import numpy as np

class Filter(ABC):
    """ 
    Base class for the Kalman and Particle filters."""
    
    def __init__(self, model, mu0, V0, T):
        """ 
        Initialises the storage arrays for the posterior distribution at each 
        time-step.
        
        Args:
            model: (Model object) describes the problem.
            mu0  : (array) contains the initial prior mean.
            V0   : (array) contains the initial prior variance.
            T    : (int) no. of timesteps.
        """
        
        self.model = model
        self.mu0 = mu0 
        self.V0 = V0
        self.T = T
        
        # Initialise storage arrays for posterior distributions.
        self.mu = np.zeros((T, *mu0.shape))
        self.V  = np.zeros((T, *V0.shape))
        
        # First entry is the initial prior distribution.
        self.mu[0, ...] = mu0
        self.V[0, ...]  = V0
        
        # Track the marginal log-likelihood.
        self.logli = 0.
    
    @abstractmethod 
    def run(self):
        pass
    
    @abstractmethod 
    def updateLogLikelihood(self):
        """ 
        This should be abstract mehtod but cba with different args"""
        
    def likelihood(self, muP, y):
        
        pass
    
class KalmanFilter(Filter):
    
    def __init__(self, model, mu0, V0, T):
        """ 
        Args:
            model: (Model object) describes the problem.
            mu0  : (array) contains the initial prior mean.
            V0   : (array) contains the initial prior variance.
            T    : (int) no. of timesteps.
        """
        super().__init__(model, mu0, V0, T)
        
        
    def predictionStep(self, mu0, V0):
        """  
        Returns the prediction distribution.
        """
        muP = self.model.A @ mu0
        VP = self.model.A @ V0 @ self.model.A.T + self.model.Q
        return muP, VP
        
    def updateStep(self, muP, VP, y):
        """ 
        Returns the updated posterior distribution for the latest time-step.
        
        Args:
            muP: the mean from the prediction step.
            VP : the covariance from the prediction step.
            y  : model observation.
        """
        S = self.model.H @ VP @ self.model.H.T + self.model.R 
        K = VP @ self.model.H.T @ np.linalg.inv(S)
        mu = muP + K @ (y - self.model.H @ muP)
        V = VP - K @ self.model.H @ VP 
        return mu, V        
    
    def run(self):
        """ 
        """
    
        mu0 = self.mu0.copy()
        V0 = self.V0.copy()
        
        for t in range(1, self.T):
            
            # This will be unique to each filter.
            mu, V = self.stepFilter(mu0, V0)
            
            # Update model and posterior.
            self.model.step()
            mu0 = mu.copy()
            V0 = V.copy()
            
            # Store latest update.
            self.mu[t, ...] = mu0
            self.V[t, ...] = V0
    
    def stepFilter(self, mu0, V0):
        """ 
        Posterior distribution calculation for the kalman filter.
        """
        
        # Prediction step to get the updated prior distribution.
        muP, VP = self.predictionStep(mu0, V0)
        
        # Get new observation.
        y = self.model.getObservation()
        
        # Update step to get the posterior distribution.
        mu, V = self.updateStep(muP, VP, y)
        
        self.updateLogLikelihood(muP, VP, y)        
        
        return mu, V
    
    def updateLogLikelihood(self, muP, VP, y):
        """
        Some code repetition bc lazy.
        """
        S = self.model.H @ VP @ self.model.H.T + self.model.R
        residual = y - self.model.H @ muP
        
        # Compute the log-likelihood contribution for the current timestep
        logli = -0.5*(residual.T @ np.linalg.inv(S) @ residual + 
                      np.log(np.linalg.det(2 * np.pi * S)))
        
        # Update stored marginal log-likelihood.
        self.logli += logli
        
class ParticleFilter(Filter):
    
    def __init__(self, model, mu0, V0, T, numParticles=1000):
        """ 
        Args:
            model: (Model object) describes the problem.
            mu0  : (array) contains the initial prior mean.
            V0   : (array) contains the initial prior variance.
            T    : (int) no. of timesteps.
        """
        super().__init__(model, mu0, V0, T)
        
        # Args unique to particle filter.
        self.numParticles = numParticles
    
    def propagateParticles(self, particles):
        """ 
        """
        noise = np.random.multivariate_normal(np.zeros(self.model.Q.shape[0]), 
                                              self.model.Q, self.numParticles)
        particlesP = self.model.A @ particles.T + noise.T
        return particlesP.T
    
    def updateStep(self, particlesP, logweights, weights):
        """ 
        """
        for i in range(self.numParticles):
            pass
    
    def run(self):
        """ 
        """
    
        # Initialise particles and weights.
        particles = np.random.multivariate_normal(self.mu0, self.V0, self.numParticles)
        logweights = np.zeros(self.numParticles)
        
        for t in range(1, self.T):
            
            # Prediction step - propagate particles through system.
            particlesP = self.propagateParticles(particles)
            
            # Get new observation.
            y = self.model.getObservation()
            
            # Update step and weight normalisation.
            for i in range(self.numParticles):
                diff = y - self.model.H @ particlesP[i]
                logweights[i] -= 0.5*diff.T @ np.linalg.inv(self.model.R) @ diff
            
            weights = np.exp(logweights - np.max(logweights))
            weights = weights / np.sum(weights)
            
            # Resampling step.
            particles = particlesP[np.random.choice(self.numParticles, 
                                                    self.numParticles, p=weights)]
            
            # Estimate and store the new posterior.
            self.mu[t, ...] = np.sum(weights[:, np.newaxis] * particlesP, axis=0)
            self.updateLogLikelihood(particlesP, y)
            
            # Update model and reset log weights.
            self.model.step()
            logweights = np.zeros(self.numParticles)
            
    def updateLogLikelihood(self, particlesP, y):
        """ 
        Bit of a nightmare.
        """
       
        # Compute log g(y_t | x_t^{(i)}) for each particle
        loglis = np.array([
            -0.5 * (y - self.model.H @ particle).T @ np.linalg.inv(self.model.R) @ 
            (y - self.model.H @ particle) - 
            0.5 * np.log(np.linalg.det(2 * np.pi * self.model.R)) 
            for particle in particlesP
        ])
        
        # Use the log-sum-exp trick for numerical stability
        logli = np.max(loglis) + np.log(
            np.sum(np.exp(loglis - np.max(loglis)))
        ) - np.log(self.numParticles)
        
        # Update the cumulative log-likelihood
        self.logli += logli