import numpy as np
import numpy.linalg as LA
# import GPy
import sys
sys.path.append(r"casadiinstalldir")
from casadi import *

class GP:
    def __init__(self, model):
        # Initialize by GPy model or by dictionary containing all information
        if type(model) == dict:
            self.rbf_variance = model["rbf_variance"]
            self.rbf_lengthscale = model["rbf_lengthscale"]
            self.noise_variance = model["noise_variance"]
            self.X = model["X_train"]
            self.Y = model["Y_train"]  
            
        self.N = len(self.Y)    
        self.lambda_inv = np.diag(1/self.rbf_lengthscale**2)
        self.Sigma_inverse()
        
    def Sigma_inverse(self):
        # Pre-compute (K_N + sigma^2*I)^-1
        K_N = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                K_N[i,j] = self.se_kernel(self.X[i,:], self.X[j,:])
        self.Sigma_inv = LA.inv(K_N + self.noise_variance**2*np.eye(self.N))
    
    def se_kernel(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
             return self.rbf_variance**2*np.exp(-LA.multi_dot([(x-y).transpose(), \
                      self.lambda_inv, (x-y)])/2)
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.rbf_variance**2*np.exp(-mtimes([(x-y).T, self.lambda_inv, (x-y)])/2)
                
    def predict(self, p):
        if type(p) == casadi.SX:
            K_star = self.se_kernel(p, p)
            K_star_N = SX.zeros(1, self.N)
            for i in range(self.N):
                K_star_N[:,i] = self.se_kernel(p, self.X[i,:])
            K_N_star = K_star_N.T 
            mean = mtimes([K_star_N, gp.Sigma_inv, self.Y])
            variance = K_star - mtimes([K_star_N, self.Sigma_inv, K_N_star]) + self.noise_variance**2         
        
        elif type(p) == np.ndarray:
            K_star = self.se_kernel(p, p)
            K_star_N = np.empty((1, self.N))
            for i in range(self.N):
                K_star_N[:,i] = self.se_kernel(p, self.X[i,:])
            K_N_star = K_star_N.transpose()  
            mean = LA.multi_dot([K_star_N, self.Sigma_inv, self.Y])
            variance = K_star - LA.multi_dot([K_star_N, self.Sigma_inv, K_N_star]) \
                + self.noise_variance**2

        return mean, variance
        

class sparse_GP:
    def __init__(self, model):
        # Initialize by GPy model or by dictionary containing all information
        if type(model) == dict:
            self.rbf_variance = model["rbf_variance"]
            self.rbf_lengthscale = model["rbf_lengthscale"]
            self.noise_variance = model["noise_variance"]
            self.X = model["X_train"]
            self.Y = model["Y_train"]
            self.Z = model["X_ind"]
            
        self.N = len(self.Y)
        self.N_ind = len(self.Z)
        self.lambda_inv = np.diag(1/self.rbf_lengthscale**2)
        self.Sigma_inverse()
        
    def Sigma_inverse(self):
        # Pre-compute (K_N + sigma^2*I)^-1        
        self.K_NZ = np.empty((self.N, self.N_ind)) 
        self.K_ZZ = np.empty((self.N_ind, self.N_ind))  
        K_NN = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N_ind):
                self.K_NZ[i,j] = self.se_kernel(self.X[i,:], self.Z[j,:])
        for i in range(self.N_ind):
            for j in range(self.N_ind):
                self.K_ZZ[i,j] = self.se_kernel(self.Z[i,:], self.Z[j,:])
        for i in range(self.N):
            for j in range(self.N):
                K_NN[i,j] = self.se_kernel(self.X[i,:], self.X[j,:])
           
        self.K_ZN = self.K_NZ.transpose()
        Q_NN = LA.multi_dot([self.K_NZ, LA.inv(self.K_ZZ), self.K_ZN])
        LAMBDA = np.diag(np.diag(K_NN - Q_NN + self.noise_variance**2*np.eye(self.N)))
        self.Sigma_inv = LA.inv(Q_NN + LAMBDA)
    
    def se_kernel(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
             return self.rbf_variance**2*np.exp(-LA.multi_dot([(x-y).transpose(), \
                      self.lambda_inv, (x-y)])/2)
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.rbf_variance**2*np.exp(-mtimes([(x-y).T, self.lambda_inv, (x-y)])/2)
                
    def predict(self, p):
        if type(p) == casadi.SX:
            K_star_Z = SX.zeros(1, self.N_ind)
            for i in range(self.N_ind):
                K_star_Z[:,i] = self.se_kernel(p, self.Z[i,:])
            Q_star_N = mtimes([K_star_Z, LA.inv(self.K_ZZ), self.K_ZN])  
            mean = mtimes([Q_star_N, self.Sigma_inv, self.Y])
            K_star = self.se_kernel(p, p)
            Q_N_star = Q_star_N.T
            variance = K_star - mtimes([Q_star_N, self.Sigma_inv, Q_N_star]) \
                + self.noise_variance**2       
        
        elif type(p) == np.ndarray:
            K_star_Z = np.empty((1, self.N_ind))
            for i in range(self.N_ind):
                K_star_Z[:,i] = self.se_kernel(p, self.Z[i,:])
            Q_star_N = LA.multi_dot([K_star_Z, LA.inv(self.K_ZZ), self.K_ZN])  
            mean = LA.multi_dot([Q_star_N, self.Sigma_inv, self.Y])
            K_star = self.se_kernel(p, p)
            Q_N_star = Q_star_N.transpose()
            variance = K_star - LA.multi_dot([Q_star_N, self.Sigma_inv, Q_N_star]) \
                + self.noise_variance**2

        return mean, variance
        