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

m1_0 = torch.zeros((m,1), dtype=torch.float32).to(dev)
m2_0 = 0 * 0 * torch.eye(m).to(dev)

# Number of Training Examples
N_E = 100

# Number of Cross Validation Examples
N_CV = 10

# NUmber of test examples
N_T = 25


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
        self.mle_K_est_out = np.array([])
        self.KNet_est_out = np.array([])
        self.test_target = np.array([])
        self.folderName = "C:/Users/sgbhanlo/Documents/KalmanNet_TSP-main/KNetFiles"
        self.seed = seed if seed > 0 else 0
        self.mean_ini = mean_ini
        self.dataFilename = self.folderName + "MCsim_test_data"
        
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
            self
            ) -> None:
        self.generateTrajectory()
        
        
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
        
        print(f"loading data and generating R and Q estimate")
        [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
        
        R_sum = np.array(np.zeros((2,2)))
        X_diff_Q_sum = np.array(np.zeros((4,4)))
        
        n_residuals = 0.0 
                
        for s in range(0, training_target.shape[0]):
            
            X_true_gen = training_target[s,:,:].numpy()
            measurements_gen = training_input[s,:,:].numpy()
            
            for k in range(1, training_target.shape[2]): # starts from 1 because measurements start from 1
            
                X_prev = X_true_gen[:, k-1] # for first itereation, state at 0
                     
                Xik = X_true_gen[:, k] # prediction at 1
        
                X_diff = Xik - F @ X_prev # State Error 1-0
                
                X_diff_mat = np.asmatrix(X_diff)
                    
                X_diff_Q_sum += X_diff_mat.T @ X_diff_mat # for the Covariance of the vectors
                
                z_diff = (measurements_gen[:,k])-(H @ Xik) # Prediction error
                
                z_diff_mat = np.asmatrix(z_diff)
                
                R_sum += (z_diff_mat.T @ z_diff_mat) # for the Covariance of Prediction error
        
                n_residuals += 1
                
        print(f"n_residuals = {n_residuals}, for R_est and Q_est")
        
        self.R_est = (1.0/ n_residuals) *  R_sum # cov prediction error
        self.Q_est = (1.0/ n_residuals) *  X_diff_Q_sum #cov state error
        print(f"R_est = {self.R_est}, Q_est = {self.Q_est}")
    
    def generateTrajTrainKNet(
            self
            ) -> None:
        
        print("generating sys model")
        self.runModelKNet(True)
    
    def loadTrajTestKNet(
            self
            ) -> None:
        
        print("testing sys model")
        self.runModelKNet(False)
        
    def runModelKNet(
            self,
            modelTrain: bool
            ) -> None:
        
        print(f"running model training with modelTrain = {modelTrain} ")
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
        KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=30, learningRate=1E-3, weightDecay=1E-5)
        
        #Generate Training and validation sequences
        
        if modelTrain: 
            print("generating data and training model")
            DataGen(sys_model, self.dataFilename, t, t)
            [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
            KNet_Pipeline.NNTrain(N_E, training_input, training_target, N_CV, cv_input, cv_target)
            KNet_Pipeline.save()
            print(f"saved KNET pipeline-trained model, saved trained modelFileName = {KNet_Pipeline.modelFileName}")
        else:
            print(f"loading data and pretrained modelFileName = {KNet_Pipeline.modelFileName} and performing test")
            [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
            KNet_Pipeline.model = torch.load(KNet_Pipeline.modelFileName)            
            [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
        
        
    def applyKalmanFilterMaxLk(
            self
            ) -> None:
  
        print(f"loading data and applying KalmanFilter")
        [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
        
        n_points = 0.0
        
        print('test target shape', test_target.shape)
        
        kalman_est = np.zeros((test_target.shape[0], 4, self.nSteps))   # state estimate
        
        
        for s in range(0, test_target.shape[0]):
            
            x_k = mean_ini # initial state estimate
            x_k_o = mean_ini
            P_k = P_ini # initial covariance estimate
            P_k_o = P_ini

            kalman_est_o = np.zeros((4, self.nSteps))
            
            X_true_gen = test_target[s,:,:].numpy()
            #print('x_treue_gen', X_true_gen)
            measurements_gen = test_input[s,:,:].numpy()
            
            
            #print('test input shape', test_input.shape)
            for k in range(0, test_target.shape[2]):    
                # Update step
                K_gain = P_k @ H.T @ np.linalg.inv( (H @ P_k @ H.T) + self.R_est)
                K_gain_orig = P_k_o @ H.T @ np.linalg.inv( (H @ P_k_o @ H.T) + R)
                #print('KGain', K_gain)
                
                x_u = x_k + K_gain @ (measurements_gen[:, k] - H @ x_k) # updated state estimate
                x_u_orig = x_k_o + K_gain_orig @ (measurements_gen[:, k] - H @ x_k_o)
                
                P_u = (np.eye(4) - K_gain @ H) @ P_k # updated covariance estimate
                P_u_o = (np.eye(4) - K_gain_orig @ H) @ P_k_o
                
                kalman_est[s, :, k] = x_u # Filtered estimate kalman_est = x_u
                kalman_est_o[:, k] = x_u_orig
                        
                
                # Predict step
            
                x_k = F @ x_u      # predicted state estimate
                P_k = F @ P_u @ F.T + self.Q_est      # predicted covariance estimate
                
                x_k_o = F @ x_u_orig # predicted state estimate
                P_k_o = F @ P_u_o @ F.T + Q # predicted covariance estimate
                
                n_points+=1.0
            
        self.mle_K_est_out = kalman_est
        self.test_target = test_target.numpy()
        print('mle_K_est_out shape', self.mle_K_est_out.shape)
        print('test target shape mle', self.test_target.shape)
        
            
        #print('kalman est', kalman_est)
        print(f"loading data and applying KalmanFilter - DONE, n_points = {n_points}")
            
    
            
    def applyKalmanFilterKNet(
            self
            ) -> None:
  
        print("applying Kalman Filter KNET")
        KNet_Pipeline = Pipeline_KF(self.strTime, self.folderName, "sysmodel")
        KNet_Pipeline = torch.load(KNet_Pipeline.PipelineName)
        [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
        print("KNet_Pipeline=", KNet_Pipeline)
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, x_out_test] = KNet_Pipeline.NNTest(test_target.shape[0], test_input, test_target) 
        #print('MSEavg', KNet_Pipeline.MSE_test_linear_arr)
        #print('MSEavgShape', KNet_Pipeline.MSE_test_linear_arr.shape)
    
        self.KNet_est_out = x_out_test.detach().numpy()
        self.test_target = test_target.numpy()
        print("Kalman Filter KNET done shape", self.KNet_est_out.shape)
        
    def computeMSEsForSequences(
            self,
            X_True: np.array,
            X_gen: np.array,
            ) -> np.array:
        
        """
        creates MSE values based on residuals between ground truths and output sequences
        """
        
        mse_T = np.zeros(X_True.shape[2])
        print('mse_T initialised with shape = ', mse_T.shape)
        
        for t in range(0, X_True.shape[2]):
        
            resid_squaresum = 0.0
            
            for s in range(0, X_True.shape[0]):
             
                X_True_ssvec = X_True[s,:,t]
                X_gen_ssvec = X_gen[s,:,t]
                
                resid_x = X_gen_ssvec[0] - X_True_ssvec[0]
                resid_y = X_gen_ssvec[2] - X_True_ssvec[2]
                 
                resid_squaresum += resid_x**2 + resid_y**2
         
            rmse = np.sqrt( resid_squaresum / X_True.shape[0]) 
            
            mse_T[t] = rmse 
        
        return mse_T
        
    
        #print(f"rmse:\n{rmse}")
       # plt.plot(range(2, n_steps+1), rmse[1:n_steps])

    
    def computeMSEMaxLk(
            self
            ) -> np.array:
        
        """
        computes MSE values for Maximum liklihood method
        """
        return self.computeMSEsForSequences(self.test_target, self.mle_K_est_out )
        
        
    def computeMSEKNet(
            self
            ) -> np.array:
        
        """
        computes MSE values for KNet method
        """
        
        return self.computeMSEsForSequences(self.test_target, self.KNet_est_out)
        