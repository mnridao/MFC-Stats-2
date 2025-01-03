# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:46:25 2025

@author: mn1215
"""

import numpy as np

class Model:
    """ 
    Class that represents the underlying system."""
    
    def __init__(self, x0, A, H, Q, R, rng=0):
        """
        Stores the model parameters.
        
        Args:
            x0: (array) contains the initial prior state.
            V0: (array) contains the initial prior covariance.
            A : (array) the state transition matrix.
            H : (array) the observation matrix. 
            Q : (array) process noise covariance. 
            R : (array) observation noise covariance.
        """
        
        # Set seed for reproducibility.
        self.rng = np.random.default_rng(rng)
        
        # Store the current state of the model (updated each time-step).
        self.currentStateX = x0 
        
        # System information.
        self.A = A
        self.H = H
        self.Q = Q 
        self.R = R
        
        self.x = [self.currentStateX]
        self.y = []
                 
    def generateModelData(self, T):
        """ 
        Generates arrays of T model states and observations."""
        
        # Initialise states array.
        # self.x = np.zeros((T, *self.currentStateX.shape))
        self.x = [self.currentStateX]
        
        # This will be populated here and is fixed.
        self.y = np.zeros((T, self.H.shape[0]))
        for i in range(T):
            self.y[i, ...] = self.getObservation()
            
    def getObservation(self):
        """ 
        Generates a new observation for the current state."""
        
        noise = self.rng.multivariate_normal(np.zeros(self.R.shape[0]), self.R)
        observation = self.H @ self.currentStateX + noise
        
        # Store.
        self.y.append(observation)
        return observation
    
    def step(self):
        """ 
        Updates the state of the underlying system."""
        noise = self.rng.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)
        self.currentStateX = self.A @ self.currentStateX + noise
        
        # Store the updated step.
        self.x.append(self.currentStateX) # yucky
            
    def getPastStates(self):
        return np.array(self.x)
    
    def getPastObservations(self):
        return np.array(self.y)
    
class ToyModelConfig:
    """ 
    Stores the data for the toy model. I would love to know a better way of 
    doing this."""
    
    def __init__(self, k, T):
        
        self.k = k
        self.T = T
        
        # Seed for reproducibility.
        self.rng = 0
                
        # State transition matrix (A) and noise covariance (Q).
        self.A = np.block([
            [np.eye(2), self.k * np.eye(2)],
            [np.zeros((2, 2)), 0.99 * np.eye(2)]
            ])
        
        self.Q = np.block([
            [self.k**3 / 3 * np.eye(2), self.k**2 / 2 * np.eye(2)],
            [self.k**2 / 2 * np.eye(2), self.k * np.eye(2)]
            ])
        
        # Observation matrix (H) and noise covariance (R).
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
            ])
    
        self.R = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
            ])
        
        # Initial prior distribution.
        self.mu0 = 3*np.ones(4)
        self.V0 = 10*np.eye(4)
        
        # Initial state.
        self.x0 = np.zeros(4)
    