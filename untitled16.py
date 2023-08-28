# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:07:43 2023

@author: sgbhanlo
"""

# -*- coding: utf-8 -*-
"""
MC simulation class
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from Linear_sysmdl import SystemModel
from Pipeline_KF import Pipeline_KF
from KalmanNet_nn import KalmanNetNN
from Extended_data import DataGen, DataLoader

mean_ini = np.array([0, 2, 0, 2])
P_ini = np.diag([1, 0.1, 1, 0.1])
chol_ini = np.linalg.cholesky(P_ini)


# Nearly constant velocity model
T = 0.5 # sampling time
F = np.array([[1, T, 0, 0], 
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]]) # state transition matrix
sigma_u = 0.5 # standard deviation of the acceleration
Q = np.array([[T**3/3, T**2/2, 0, 0], 
              [T**2/2, T, 0, 0],
              [0, 0, T**3/3, T**2/2],
              [0, 0, T**2/2, T]]) * sigma_u**2 # covariance of the process noise

chol_Q = np.linalg.cholesky(Q) # Cholesky decomposition of Q
r = 1.0 #rms of standard normal dist

# Measurement model
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]]) # measurement matrix
R = 0.5 * np.diag([1, 1]) # covariance of the measurement noise
chol_R = np.linalg.cholesky(R) # Cholesky decomposition of R

##initisalisng space on device (CPU)
if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

m = 4
n = 2

m1_0 = torch.tensor([[0.0], [0.0]]).to(dev)
#m1_0 = torch.tensor([0, 2, 0, 2], dtype=torch.float32).to(dev)
m2_0 = 0 * 0 * torch.eye(m).to(dev)

# Number of Training Examples
N_E = 1000

# Number of Cross Validation Examples
N_CV = 100

N_T = 200


class MonteCarloSimulation:
    def __init__(
            self,
            nSteps: int, 
            seed: int = 0,
            ) -> None:
        self.nSteps = nSteps
        self.X_true = np.array([])
        self.measurements = np.array([])
        self.Q_est = np.array([])
        self.R_est = np.array([])
        self.x_k = np.array([])
        self.x_k_o = np.array([])
        self.P_k = np.array([])
        self.P_k_o = np.array([])
        self.X_diff_Q_sum = np.array([])
        self.kalman_est = np.array([])
        self.kalman_est_o = np.array([])
        self.folderName = "C:/Users/sgbhanlo/Documents/KalmanNet_TSP-main/KNetFiles"
        self.seed = seed if seed > 0 else 0
        self.mean_ini = mean_ini
        
        today = datetime.today()
        now = datetime.now()
        strToday = today.strftime("%m.%d.%y")
        strNow = now.strftime("%H:%M:%S")
        self.strTime = strToday + "_" + strNow
        print("Current Time =", self.strTime)
        print('Initialised MC object')

    def generateTrajectory(
            self, 
            meanIni: np.array = None
            ) -> None:
        
        if self.seed > 0:
            np.random.seed(self.seed)
            print('Using random seed', self.seed, "reset seed to 0")
            self.seed = 0
            
        self.X_true = np.zeros((4, self.nSteps)) # state vector (x, x_vel, y, y_vel)
        self.measurements = np.zeros((2, self.nSteps))
        self.X_diff_Q_sum = np.array(np.zeros((4,4)))
    
        # Generate ground truth
        if meanIni is None:
            meanIni = self.mean_ini
        
        print("MeanIni", meanIni)
        self.X_true[:, 0] = meanIni + np.dot(chol_ini, np.random.randn(4))
        
        # Generate measurements
        self.measurements[:, 0] = np.dot(H,  self.X_true[:, 0]) + np.dot(chol_R, np.random.randn(2)) 
        
        # Generate the complete trajectory and measurements
        for k in range(1, self.nSteps):
            # Propagate the true state
            self.X_true[:, k] = np.dot(F,  self.X_true[:, k-1]) + np.dot(chol_Q, np.random.randn(4))
            
            # Generate measurement (adding noise with rms = 1, see r above)
            self.measurements[:, k] = np.dot(H, self.X_true[:, k]) + np.dot(chol_R, np.random.randn(2))
     
    def generateSequenceTorch(
            self,
            meanIniTorch: torch.tensor = None
            ) -> None:
        if meanIniTorch is not None:
            self.generateTrajectory(meanIniTorch.numpy())
        else:
            self.generateTrajectory(meanIniTorch)
        
        
    def getTrajectoryArrays(
            self
            ) -> dict:
        return {
                "X_true": self.X_true, 
                "measurements": self.measurements
                }      
    
    def getTrajectoryArraysTorch(
            self
            ) -> dict:
        X_true_torch = torch.tensor(self.X_true, dtype = torch.float32)
        measurements_torch = torch.tensor(self.measurements, dtype = torch.float32)
        return {
                "X_true": X_true_torch, 
                "measurements": measurements_torch
                }
        
    def generateEstimatesMaxLk(
            self
            ) -> None:
        
        R_sum = np.array(np.zeros((2,2)))
        X_diff_Q_sum = np.array(np.zeros((4,4)))
        
        for k in range(1,self.nSteps): # starts from 1 because measurements start from 1
            X_prev = self.X_true[:, k-1] # for first itereation, state at 0
                 
            Xik = self.X_true[:, k] # prediction at 1
    
            X_diff = Xik - F @ X_prev # State Error 1-0
            
            X_diff_mat = np.asmatrix(X_diff)
                
            X_diff_Q_sum += X_diff_mat.T @ X_diff_mat # for the Covariance of the vectors
            
            z_diff = (self.measurements[:,k])-(H @ Xik) # Prediction error
            
            z_diff_mat = np.asmatrix(z_diff)
            
            R_sum += (z_diff_mat.T @ z_diff_mat) # for the Covariance of Prediction error
            
        self.R_est = (1/(self.nSteps)) *  R_sum # cov prediction error
        self.Q_est = (1/(self.nSteps)) *  X_diff_Q_sum #cov state error
    
    
    def generateEstimatesKNet(
            self
            ) -> None:
        
        print("generating sys model")
        t = self.nSteps #to align with sys model T
        
        sys_model = SystemModel(
            torch.tensor(F, dtype = torch.float32),
            torch.tensor(Q, dtype = torch.float32),
            torch.tensor(H, dtype = torch.float32),
            torch.tensor(r, dtype = torch.float32),
            t, t)
        sys_model.InitSequence(m1_0, m2_0)
        sys_model.SetTrajectoryGenerator(self)
        
        
        print("Start KNet pipeline, KNet with full model info")
        KNet_Pipeline = Pipeline_KF(self.strTime, self.folderName, "sysmodel")
        KNet_Pipeline.setssModel(sys_model)
        KNet_model = KalmanNetNN()
        KNet_model.Build(sys_model)
        KNet_Pipeline.setModel(KNet_model)
        KNet_Pipeline.setTrainingParams(n_Epochs=20, n_Batch=30, learningRate=1E-3, weightDecay=1E-5)
        
        #Generate Training and validation sequences
        
        dataFilename = self.folderName + "MCsim_test_data"
        DataGen(sys_model, dataFilename, t, t)
        [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFilename)
       
        KNet_Pipeline.NNTrain(N_E, training_input, training_target, N_CV, cv_input, cv_target)
        [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
        KNet_Pipeline.save()
        print("saved KNET pipeline")
            
        
    def applyKalmanFilterMaxLk(
            self
            ) -> None:
  
        self.x_k = mean_ini # initial state estimate
        self.x_k_o = mean_ini
        self.P_k = P_ini # initial covariance estimate
        self.P_k_o = P_ini

        self.kalman_est = np.zeros((4, self.nSteps)) # state estimate
        self.kalman_est_o = np.zeros((4, self.nSteps))
        
        for k in range(0, self.nSteps):    
            # Update step
            K_gain = self.P_k @ H.T @ np.linalg.inv( (H @ self.P_k @ H.T) + self.R_est)
            K_gain_orig = self.P_k_o @ H.T @ np.linalg.inv( (H @ self.P_k_o @ H.T) + R)
            
            x_u = self.x_k + K_gain @ (self.measurements[:, k] - H @ self.x_k) # updated state estimate
            x_u_orig = self.x_k_o + K_gain_orig @ (self.measurements[:, k] - H @ self.x_k_o)
            
            P_u = (np.eye(4) - K_gain @ H) @ self.P_k # updated covariance estimate
            P_u_o = (np.eye(4) - K_gain_orig @ H) @ self.P_k_o
            
            self.kalman_est[:, k] = x_u # Filtered estimate kalman_est = x_u
            self.kalman_est_o[:, k] = x_u_orig
                    
            # Predict step
    
            self.x_k = F @ x_u      # predicted state estimate
            self.P_k = F @ P_u @ F.T + self.Q_est      # predicted covariance estimate
            
            self.x_k_o = F @ x_u_orig # predicted state estimate
            self.P_k_o = F @ P_u_o @ F.T + Q # predicted covariance estimate
            