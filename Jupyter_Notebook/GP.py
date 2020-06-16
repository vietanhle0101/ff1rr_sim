import numpy as np
import numpy.linalg as LA
import GPy
import sys
sys.path.append(r"casadiinstalldir")
from casadi import *


class GP:
    def __init__(self, model):
        # Initialize by GPy model or by dictionary containing all information
        if type(model) == GPy.models.gp_regression.GPRegression:
            self.rbf_variance = model.rbf.variance.values[0]
            self.rbf_lengthscale = model.rbf.lengthscale.values
            self.noise_variance = model.Gaussian_noise.variance.values[0]
            self.X = np.array(model.X)
            self.Y = model.Y
        elif type(model) == dict:
            self.rbf_variance = model["rbf_variance"]
            self.rbf_lengthscale = model["rbf_lengthscale"]
            self.noise_variance = model["noise_variance"]
            self.X = model["X_train"]
            self.Y = model["Y_train"]  
            
        self.N = len(self.Y)    
        self.n = np.shape(self.X)[1]
        self.lambda_inv = np.diag(1/self.rbf_lengthscale**2)
        
        x = SX.sym('x', self.n, 1)
        y = SX.sym('y', self.n, 1)
        k_xy = self.rbf_variance**2*exp(-0.5*mtimes([(x-y).T, self.lambda_inv, (x-y)]))
        self.k = Function('k', [x, y], [k_xy])
        
        self.Sigma_inverse()
        
    def Sigma_inverse(self):
        # Pre-compute (K_N + sigma^2*I)^-1 and (K_N + sigma^2*I)^-1*Y
        K_N = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                K_N[i,j] = self.se_kernel(self.X[i,:], self.X[j,:])
        self.Sigma_inv = LA.inv(K_N + self.noise_variance**2*np.eye(self.N))
        self.alpha = np.dot(self.Sigma_inv, self.Y) 
    
    def se_kernel(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k(x, y)
                
    def predict(self, p):
        if type(p) == casadi.SX:
            K_star = self.se_kernel(p, p)
            K_xN = self.k.map(self.N) 
            K_star_N = K_xN(p, self.X.transpose())
            K_N_star = K_star_N.T 
            mean = mtimes([K_star_N, self.alpha])
            variance = K_star - mtimes([K_star_N, self.Sigma_inv, K_N_star]) + self.noise_variance**2         
        
        elif type(p) == np.ndarray:
            K_star = self.se_kernel(p, p)
            K_xN = self.k.map(self.N) 
            K_star_N = K_xN(p, self.X.transpose()).full()
            K_N_star = K_star_N.transpose()  
            mean = LA.multi_dot([K_star_N, self.alpha])
            variance = K_star - LA.multi_dot([K_star_N, self.Sigma_inv, K_N_star]) \
                + self.noise_variance**2

        return mean, variance
        
        
class sparse_GP:
    def __init__(self, model):
        # Initialize by GPy model or by dictionary containing all information
        if type(model) == GPy.models.sparse_gp_regression.SparseGPRegression:
            self.rbf_variance = model.rbf.variance.values[0]
            self.rbf_lengthscale = model.rbf.lengthscale.values
            self.noise_variance = model.Gaussian_noise.variance.values[0]
            self.X = np.array(model.X)
            self.Y = model.Y
            self.Z = model.inducing_inputs.values
        elif type(model) == dict:
            self.rbf_variance = model["rbf_variance"]
            self.rbf_lengthscale = model["rbf_lengthscale"]
            self.noise_variance = model["noise_variance"]
            self.X = model["X_train"]
            self.Y = model["Y_train"]
            self.Z = model["X_ind"]
            
        self.N = len(self.Y)
        self.N_ind = len(self.Z)
        self.n = np.shape(self.X)[1]
        self.lambda_inv = np.diag(1/self.rbf_lengthscale**2)

        x = SX.sym('x', self.n, 1)
        y = SX.sym('y', self.n, 1)
        k_xy = self.rbf_variance**2*exp(-0.5*mtimes([(x-y).T, self.lambda_inv, (x-y)]))
        self.k = Function('k', [x, y], [k_xy])
        
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
        self.alpha = LA.multi_dot([LA.inv(self.K_ZZ), self.K_ZN, self.Sigma_inv, self.Y]) 
        self.beta = LA.multi_dot([LA.inv(self.K_ZZ), self.K_ZN, self.Sigma_inv, \
                                  self.K_ZN.transpose(), LA.inv(self.K_ZZ).transpose()]) 
    
    def se_kernel(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k(x, y)
                
    def predict(self, p):
        if type(p) == casadi.SX:
            K_xN = self.k.map(self.N_ind) 
            K_star_Z = K_xN(p, self.Z.transpose())
            mean = mtimes(K_star_Z, self.alpha)
            K_star = self.se_kernel(p, p)
            K_Z_star = K_star_Z.T
            variance = K_star - mtimes([K_star_Z, self.beta, K_Z_star]) \
                + self.noise_variance**2       
        
        elif type(p) == np.ndarray:
            K_xN = self.k.map(self.N_ind) 
            K_star_Z = K_xN(p, self.Z.transpose()).full()
            mean = np.dot(K_star_Z, self.alpha)
            K_star = self.se_kernel(p, p)
            K_Z_star = K_star_Z.transpose()
            
            variance = K_star - LA.multi_dot([K_star_Z, self.beta, K_Z_star]) \
                + self.noise_variance**2

        return mean, variance
    
    
class linsparse_GP:
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
        self.n = np.shape(self.X)[1]
        self.lambda_inv = np.diag(1/self.rbf_lengthscale**2)
        
        x = SX.sym('x', self.n, 1)
        y = SX.sym('y', self.n, 1)
        k_xy = self.rbf_variance**2*exp(-0.5*mtimes([(x-y).T, self.lambda_inv, (x-y)]))
        self.k = Function('k', [x, y], [k_xy])
        k10_xy = gradient(k_xy, x) 
        self.k10 = Function('k10', [x, y], [k10_xy])
        k01_xy = jacobian(k_xy, y)
        self.k01 = Function('k01', [x, y], [k01_xy])
        k11_xy = jacobian(k10_xy, y)
        self.k11 = Function('k11', [x, y], [k11_xy])
        
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
        self.alpha = LA.multi_dot([LA.inv(self.K_ZZ), self.K_ZN, self.Sigma_inv, self.Y]) 
        self.beta = LA.multi_dot([LA.inv(self.K_ZZ), self.K_ZN, self.Sigma_inv, \
                                  self.K_ZN.transpose(), LA.inv(self.K_ZZ).transpose()]) 
    
    def se_kernel(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k(x, y)
        
    def se_kernel_01(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k01(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k01(x, y)
                
    def se_kernel_10(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k10(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k10(x, y)
        
    def se_kernel_11(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k11(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k11(x, y)
                
    def predict(self, p):
        if type(p) == casadi.SX:
            K_xN = self.k.map(self.N_ind) 
            K_star_Z = K_xN(p, self.Z.transpose())
            mean = mtimes(K_star_Z, self.alpha)
            K_star = self.se_kernel(p, p)
            K_Z_star = K_star_Z.T
            variance = K_star - mtimes([K_star_Z, self.beta, K_Z_star]) \
                + self.noise_variance**2       
        
        elif type(p) == np.ndarray:
            K_xN = self.k.map(self.N_ind) 
            K_star_Z = K_xN(p, self.Z.transpose()).full()
            mean = np.dot(K_star_Z, self.alpha)
            K_star = self.se_kernel(p, p)
            K_Z_star = K_star_Z.transpose()
            
            variance = K_star - LA.multi_dot([K_star_Z, self.beta, K_Z_star]) \
                + self.noise_variance**2

        return mean, variance
    
    def linearize(self, p, use_var = False):
        if type(p) == np.ndarray:
            K_xN = self.k.map(self.N_ind) 
            K_star_Z = K_xN(p, self.Z.transpose())
            K10_xN = self.k10.map(self.N_ind) 
            K10_star_Z = K10_xN(p, self.Z.transpose())
            A = np.vstack([K_star_Z, K10_star_Z])
            mx = np.dot(A, self.alpha)
            if use_var: 
                K_star = self.se_kernel(p, p)
                K01_star = self.se_kernel_01(p, p)
                K10_star = self.se_kernel_10(p, p)
                K11_star = self.se_kernel_11(p, p)

                B = np.vstack([np.hstack([K_star, K01_star]), np.hstack([K10_star, K11_star])])
                C = LA.multi_dot([A, self.beta, A.T])
                Vx = B-C            
                return mx, Vx
            else: return mx
        
        
class linGP:
    def __init__(self, model):
        # Initialize by GPy model or by dictionary containing all information
        if type(model) == dict:
            self.rbf_variance = model["rbf_variance"]
            self.rbf_lengthscale = model["rbf_lengthscale"]
            self.noise_variance = model["noise_variance"]
            self.X = model["X_train"]
            self.Y = model["Y_train"]  
            
        self.N = len(self.Y)
        self.n = np.shape(self.X)[1]
        self.lambda_inv = np.diag(1/self.rbf_lengthscale**2)
        
        x = SX.sym('x', self.n, 1)
        y = SX.sym('y', self.n, 1)
        k_xy = self.rbf_variance**2*exp(-0.5*mtimes([(x-y).T, self.lambda_inv, (x-y)]))
        self.k = Function('k', [x, y], [k_xy])
        k10_xy = gradient(k_xy, x) 
        self.k10 = Function('k10', [x, y], [k10_xy])
        k01_xy = jacobian(k_xy, y)
        self.k01 = Function('k01', [x, y], [k01_xy])
        k11_xy = jacobian(k10_xy, y)
        self.k11 = Function('k11', [x, y], [k11_xy])
        
        self.Sigma_inverse()
        
    def Sigma_inverse(self):
        # Pre-compute (K_N + sigma^2*I)^-1 and (K_N + sigma^2*I)^-1*Y
        K_N = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                K_N[i,j] = self.se_kernel(self.X[i,:], self.X[j,:])
        self.Sigma_inv = LA.inv(K_N + self.noise_variance**2*np.eye(self.N))
        self.alpha = np.dot(self.Sigma_inv, self.Y)
    
    def se_kernel(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k(x, y)
        
    def se_kernel_01(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k01(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k01(x, y)
                
    def se_kernel_10(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k10(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k10(x, y)
        
    def se_kernel_11(self, x, y):
        if type(x) == np.ndarray and type(y) == np.ndarray:
            return self.k11(x, y).full()
        elif type(x) == casadi.SX or type(y) == casadi.SX:
            return self.k11(x, y)
                
    def predict(self, p):
        if type(p) == casadi.SX:
            K_star = self.se_kernel(p, p)
            K_xN = self.k.map(self.N) 
            K_star_N = K_xN(p, self.X.transpose())
            K_N_star = K_star_N.T 
            mean = mtimes([K_star_N, self.alpha])
            variance = K_star - mtimes([K_star_N, self.Sigma_inv, K_N_star]) + self.noise_variance**2         
        
        elif type(p) == np.ndarray:
            K_star = self.se_kernel(p, p)
            K_xN = self.k.map(self.N) 
            K_star_N = K_xN(p, self.X.transpose()).full()
            K_N_star = K_star_N.transpose()  
            mean = LA.multi_dot([K_star_N, self.alpha])
            variance = K_star - LA.multi_dot([K_star_N, self.Sigma_inv, K_N_star]) \
                + self.noise_variance**2

        return mean, variance
    
    def linearize(self, p, use_var = False):
        if type(p) == np.ndarray:
            K_xN = self.k.map(self.N) 
            K_star_N = K_xN(p, self.X.transpose())
            K10_xN = self.k10.map(self.N)
            K10_star_N = K10_xN(p, self.X.transpose())

            A = np.vstack([K_star_N, K10_star_N])
            mx = np.dot(A, self.alpha)
            if use_var: 
                K_star = self.se_kernel(p, p)
                K01_star = self.se_kernel_01(p, p)
                K10_star = self.se_kernel_10(p, p)
                K11_star = self.se_kernel_11(p, p)

                B = np.vstack([np.hstack([K_star, K01_star]), np.hstack([K10_star, K11_star])])
                C = LA.multi_dot([A, self.Sigma_inv, A.T])
                Vx = B-C            
                return mx, Vx
            else: return mx    
        