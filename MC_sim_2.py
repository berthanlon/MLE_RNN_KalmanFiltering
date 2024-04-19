# -*- coding: utf-8 -*-
"""
MC simulation class
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from Linear_sysmdl import SystemModel
from Pipeline_KF import Pipeline_KF
from KalmanNet_nn import KalmanNetNN
from Extended_data import DataGen,DataLoader, DataLoader_GPU #, N_E, N_CV, N_T, Tlen, Tlen_test
#from parameters_2 import F_bet, H_bet, Q_bet, R_bet, Tlen, Tlen_test, chol_Q_bet, chol_R

mean_ini = np.array([0, 2, 0, 2])
P_ini = np.diag([1, 0.1, 1, 0.1])
chol_ini = np.linalg.cholesky(P_ini)

# Nearly constant velocity model
T = 0.5 # sampling time
F_bet = np.array([[1, T, 0, 0], 
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]], dtype = np.float64) # state transition matrix
sigma_u = 0.001 # standard deviation of the acceleration    ####
Q_bet = np.array([[T**3/3, T**2/2, 0, 0], 
              [T**2/2, T, 0, 0],
              [0, 0, T**3/3, T**2/2],
              [0, 0, T**2/2, T]], dtype = np.float64) * sigma_u**2 # covariance of the process noise

chol_Q = np.linalg.cholesky(Q_bet) # Cholesky decomposition of Q
r = 1.0 #rms of standard normal dist

# Measurement model
H_bet = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]], dtype = np.float64) # measurement matrix
R_bet = 0.5 * np.diag([1, 1]).astype(dtype= np.float64) # covariance of the measurement noise ###
chol_R = np.linalg.cholesky(R_bet) # Cholesky decomposition of R

##initisalisng space on device (CPU)
if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print('Running on CUDA')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")
   
r = 1
m = 4
n = 2

m1_0 = torch.zeros((m,1), dtype = torch.float64).to(dev) #dtype = torch.float32
m2_0 = 0 * 0 * torch.eye(m, dtype = torch.float64).to(dev)

print('M10,M20 DTYPE= ', m1_0.dtype, m2_0.dtype)


class MonteCarloSimulation:
    def __init__(
            self,
            nSteps: int, 
            baseDir: str,
            fileName: str,
            args: argparse.Namespace,
            seed: int = 0,
            ) -> None:
        #self.nSteps = nSteps
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
        self.folderName = baseDir
        self.seed = seed if seed > 0 else 0
        self.mean_ini = mean_ini
        self.dataFilename = self.folderName + fileName
        self.MLETrajFilename = self.folderName + fileName + '_MLETraj.npy'
        self.KNetTrajFilename = self.folderName + fileName + '_KNetTraj.npy'
        self.args = args
        
        today = datetime.today()
        now = datetime.now()
        strToday = today.strftime("%m.%d.%y")
        strNow = now.strftime("%H:%M:%S")
        self.strTime = strToday + "_" + strNow
        print("Current Time =", self.strTime)
        print('args', self.args)
        print(f'Initialised MC object, nSteps = {self.args.nSteps}, fileName = {fileName}, seed = {seed}')

    def generateTrajectory(
            self, 
            meanIni: np.array = None
            ) -> None:
        
        if self.seed > 0:
            np.random.seed(self.seed)
            print('Using random seed', self.seed, "reset seed to 0")
            self.seed = 0
            
        self.X_true = np.zeros((4, self.args.nSteps)) # state vector (x, x_vel, y, y_vel)
        self.measurements = np.zeros((2, self.args.nSteps))
        self.X_diff_Q_sum = np.array(np.zeros((4,4)))
    
        # Generate ground truth
        if meanIni is None:
            meanIni = self.mean_ini
            
        self.X_true[:, 0] = meanIni + np.dot(chol_ini, np.random.randn(4))
        
        # Generate the complete trajectory and measurements
        for k in range(1, self.args.nSteps):
            # Propagate the true state
            
            self.X_true[:, k] = np.dot(F_bet,  self.X_true[:, k-1]) + np.dot(chol_Q, np.random.randn(4))
            
            # Generate measurements
        self.measurements[:, 0] = np.dot(H_bet,  self.X_true[:, 0]) + np.dot(chol_R, np.random.randn(2)) 
            
        for k in range(1, self.args.nSteps):
            # Propagate the true state
            
            # Generate measurement (adding noise with rms = 1, see r above)
            self.measurements[:, k] = np.dot(H_bet, self.X_true[:, k]) + np.dot(chol_R, np.random.randn(2))
        
        #print('x_true_15', self.X_true[:, 1], 'shape=', self.X_true[:, 1].shape )
        #print('x_true_16', self.X_true[:, 2])
               
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
        X_true_torch = torch.tensor(self.X_true)  #dtype = torch.float32
        measurements_torch = torch.tensor(self.measurements) #dtype = torch.float32
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
            
            X_true_gen = training_target[s,:,:].cpu().numpy()
            measurements_gen = training_input[s,:,:].cpu().numpy()
            
            for k in range(1, training_target.shape[2]): # starts from 1 because measurements start from 1
            
                X_prev = X_true_gen[:, k-1] # for first itereation, state at 0
                     
                Xik = X_true_gen[:, k] # prediction at 1
        
                X_diff = Xik - F_bet @ X_prev # State Error 1-0
                
                X_diff_mat = np.asmatrix(X_diff)
                    
                X_diff_Q_sum += X_diff_mat.T @ X_diff_mat # for the Covariance of the vectors
                
                z_diff = (measurements_gen[:,k])-(H_bet @ Xik) # Prediction error
                
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
        t = self.args.nSteps #to align with sys model T
        
        sys_model = SystemModel(
            torch.tensor(F_bet), #dtype = torch.float32
            torch.tensor(Q_bet), #dtype = torch.float32
            torch.tensor(H_bet), #dtype = torch.float32
            torch.tensor(R_bet), #dtype = torch.float32
            self.args.Tlen, self.args.Tlen_test)
        sys_model.InitSequence(m1_0, m2_0)
        sys_model.SetTrajectoryGenerator(self)
             
        print("Start KNet pipeline, KNet with full model info")
        KNet_Pipeline = Pipeline_KF(self.strTime, self.folderName, f"sysmodel_{self.args.nSteps}") #"sysmodel"
        
        KNet_Pipeline.setssModel(sys_model)
        KNet_model = KalmanNetNN()
        KNet_model.NNBuild(sys_model, self.args)
        print(KNet_model)
        KNet_Pipeline.setModel(KNet_model)
        KNet_Pipeline.setTrainingParams(n_Epochs= 200, n_Batch= self.args.n_batch, learningRate= self.args.lr, weightDecay=self.args.wd)
 
        #Generate Training and validation sequences
        
        if modelTrain: 
            print("generating data and training model")
            DataGen(sys_model, self.dataFilename, t, t, self.args)
            [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(self.dataFilename)
            print('training_input type, training_target, cv_input, cv_target, test_input, test_target', [type(training_input), training_target.shape, cv_input.shape, cv_target.shape, test_input.shape, test_target.shape])
            KNet_Pipeline.NNTrain(self.args.N_E, training_input, training_target, self.args.N_CV, cv_input, cv_target)
            KNet_Pipeline.save()
            print(f"saved KNET pipeline-trained model, saved trained modelFileName = {KNet_Pipeline.modelFileName}")
        else:
            print(f"loading data and pretrained modelFileName = {KNet_Pipeline.modelFileName} and performing test")
            [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(self.dataFilename)
            KNet_Pipeline.model = torch.load(KNet_Pipeline.modelFileName,map_location=dev) 
            [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(self.args.N_T, test_input, test_target)
           
            
    def applyKalmanFilterMaxLk(
            self
            ) -> None:
  
        print(f"loading data and applying KalmanFilter")
        [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
        
        n_points = 0.0
        
        print('test target shape', test_target.shape)
        
        kalman_est = np.zeros((test_target.shape[0], 4, self.args.nSteps))   # state estimate
        
        
        for s in range(0, test_target.shape[0]):
            
            x_k = mean_ini # initial state estimate
            x_k_o = mean_ini
            P_k = P_ini # initial covariance estimate
            P_k_o = P_ini

            kalman_est_o = np.zeros((4, self.args.nSteps))
            
            X_true_gen = test_target[s,:,:].cpu().numpy()
            #print('x_treue_gen', X_true_gen)
            measurements_gen = test_input[s,:,:].cpu().numpy()
            
            
            #print('test input shape', test_input.shape)
            for k in range(0, test_target.shape[2]):    
                # Update step
                K_gain = P_k @ H_bet.T @ np.linalg.inv( (H_bet @ P_k @ H_bet.T) + self.R_est)
                K_gain_orig = P_k_o @ H_bet.T @ np.linalg.inv( (H_bet @ P_k_o @ H_bet.T) + R_bet)
                #print('KGain', K_gain)
                
                x_u = x_k + K_gain @ (measurements_gen[:, k] - H_bet @ x_k) # updated state estimate
                x_u_orig = x_k_o + K_gain_orig @ (measurements_gen[:, k] - H_bet @ x_k_o)
                
                P_u = (np.eye(4) - K_gain @ H_bet) @ P_k # updated covariance estimate
                P_u_o = (np.eye(4) - K_gain_orig @ H_bet) @ P_k_o
                
                kalman_est[s, :, k] = x_u # Filtered estimate kalman_est = x_u
                kalman_est_o[:, k] = x_u_orig
                        
                
                # Predict step
            
                x_k = F_bet @ x_u      # predicted state estimate
                P_k = F_bet @ P_u @ F_bet.T + self.Q_est      # predicted covariance estimate
                
                x_k_o = F_bet @ x_u_orig # predicted state estimate
                P_k_o = F_bet @ P_u_o @ F_bet.T + Q_bet # predicted covariance estimate
                
                n_points+=1.0
            
        self.mle_K_est_out = kalman_est
        np.save(self.MLETrajFilename, kalman_est)
        
        self.test_target = test_target.cpu().numpy()
        print('mle_K_est_out shape', self.mle_K_est_out.shape)
        print('test target shape mle', self.test_target.shape)
        
            
        #print('kalman est', kalman_est)
        print(f"loading data and applying KalmanFilter - DONE, n_points = {n_points}")
            
    
            
    def applyKalmanFilterKNet(
            self
            ) -> None:
  
        print("applying Kalman Filter KNET")
        KNet_Pipeline = Pipeline_KF(self.strTime, self.folderName, f"sysmodel_{self.args.nSteps}")
        KNet_Pipeline = torch.load(KNet_Pipeline.PipelineName,map_location=dev)
        [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
        print("KNet_Pipeline=", KNet_Pipeline)
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, x_out_test] = KNet_Pipeline.NNTest(test_target.shape[0], test_input, test_target) 
        print('MSEavg', KNet_Pipeline.MSE_test_linear_arr)
        print('MSEavgShape', KNet_Pipeline.MSE_test_linear_arr.shape)

        self.KNet_est_out = x_out_test.detach().cpu().numpy()
        np.save(self.KNetTrajFilename, self.KNet_est_out)
        
        self.test_target = test_target.cpu().numpy()
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
        square_errors_sum = np.zeros(X_True.shape[2])
        
        print('mse_T initialised with shape = ', mse_T.shape)
        print('X_gen.shape', X_gen.shape)
        
        print('xtrue shape', X_True.shape)
        
        for t in range(0, X_True.shape[2]):
        
            resid_squaresum = 0.0
            
            for s in range(0, X_True.shape[0]):
             
                X_True_ssvec = X_True[s,:,t]
                X_gen_ssvec = X_gen[s,:,t]
                
                resid_x = X_gen_ssvec[0] - X_True_ssvec[0]
                resid_y = X_gen_ssvec[2] - X_True_ssvec[2]
                 
                resid_squaresum += resid_x**2 + resid_y**2
         
            
            rmse = np.sqrt( resid_squaresum / X_True.shape[0]) 
            #totalrmse = residsquaresum
            square_errors_sum[t] = resid_squaresum
            mse_T[t] = rmse 
            #print('MSE-t[t]', mse_T)
        
        overall_rmse = np.sqrt((np.sum(square_errors_sum)) / (X_True.shape[2]*X_True.shape[0]))
        
        print('NEW SQRT OVERALL RMSE=', overall_rmse)
        
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


    def saveOutTraj(
            self,
            X_True: np.array,
            X_gen: np.array,
            ) -> None:
        
        plt.figure()
        
        print('mlextrue', X_True.shape)
        #print('xtrue shape', X_True.shape)
        for s in range(0, 9): #X_True.shape[0]):
            
            X_True_ssvec = X_True[s,:,:]
            X_gen_ssvec = X_gen[s,:,:]
            
            if s==0:
                plt.plot(X_True_ssvec[0], X_True_ssvec[2], color = 'r', label = 'ground truth')# label=f't={t}, s={s}')
                plt.plot(X_gen_ssvec[0], X_gen_ssvec[2], color = 'b', label = 'KF estimate')# label=f't={t}, s={s}')
            else:
                plt.plot(X_True_ssvec[0], X_True_ssvec[2], color = 'r')# label=f't={t}, s={s}')
                plt.plot(X_gen_ssvec[0], X_gen_ssvec[2], color = 'b')# label=f't={t}, s={s}')
            plt.legend()
            #print('trajectories', X_True_ssvec[0]) #, X_True_ssvec[2])
                
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        #plt.title('Superimposed Trajectories (True vs Generated)')
        #plt.legend()  # Show a legend with labels for each trajectory
        #folderpath = f'C:/Users/betti/Desktop/MLE_KNET/MLE_RNN_KalmanFiltering-main/KNetFiles_18/'
        #plotname
        plt.savefig('mletraj.eps', format = 'eps')
        plt.show()
        return 
    
    def plotAllTraj(
            self,
            X_True: np.array,
            measurements: np.array,
            X_gen_KNet: np.array,
            X_gen_MLE: np.array,
            ) -> None:
        
        plt.figure()
        
        #print('xTrue2 shape', X_True.shape)
        for r in range(0, X_True.shape[0]):
            
            #print('X_True=', X_True)
            #print('x_gen_KNet shaoe', X_gen_KNet.shape)
            
            X_True_ssvec = X_True[r,:,:]
            measurements_ssvec = measurements[r,:,:]
            X_gen_KNet_ssvec = X_gen_KNet[r,:,:]
            X_gen_MLE_ssvec = X_gen_MLE[r,:,:]
            
            if r==14:
                plt.plot(X_True_ssvec[0], X_True_ssvec[2], color = 'k', label = 'ground truth')# label=f't={t}, s={s}')
                plt.plot(X_gen_KNet_ssvec[0], X_gen_KNet_ssvec[2], color = 'r', label = 'KalmanNet')
                plt.plot(X_gen_MLE_ssvec[0], X_gen_MLE_ssvec[2], color = 'g', label = 'Kalman-MLE')# label=f't={t}, s={s}')
                plt.scatter(measurements_ssvec[0], measurements_ssvec[1], color = 'b', label = 'measurements', marker = 'x')
                plt.legend()
            #print('trajectories', X_True_ssvec[0]) #, X_True_ssvec[2])
                
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.axis('equal')
        plt.grid()
        #plt.title('Superimposed Trajectories (True vs Generated)')
        #plt.legend()  # Show a legend with labels for each trajectory
        #folderpath = f'C:/Users/betti/Desktop/MLE_KNET/MLE_RNN_KalmanFiltering-main/KNetFiles_18/'
        #plotname
        plt.savefig('AllTraj50_14.eps', format = 'eps')
        plt.savefig('AllTraj50_14.png', format = 'png')
        plt.show()
        return 
    
    def allTrajKNet(
            self
            ) -> None: 
        
        return self.saveOutTraj(self.test_target, self.KNet_est_out)
    
    
    def allTrajMLE(
            self
            ) -> None: 
    
        return self.saveOutTraj(self.test_target, self.mle_K_est_out)

       
    def runPlotALL(
            self
            ) -> None:
        [training_input, training_target, cv_input, cv_target, test_input, test_target] = DataLoader(self.dataFilename)
        
        KNet_est_out = np.load(self.KNetTrajFilename)
        kalman_est_mle = np.load(self.MLETrajFilename)
        test_target_arr = test_target.cpu().numpy() 
        test_input_arr = test_input.cpu().numpy()
        
        
        print('testinput shape', test_input_arr.shape)
        
        return self.plotAllTraj(test_target_arr, test_input_arr, KNet_est_out, kalman_est_mle)
    
 #   X_gen_ssvec = X_gen[s,:,t]
    
"""    
    def plotAllTrajectories(
            self,
            X_True: np.array,
            X_gen: np.array,
            ) -> None:
        for t in range(0, X_True.shape[2]):
            
            for s in range(0, X_True.shape[0]):
                
                X_True_ssvec = X_True[s,:,t]
                X_gen_ssvec = X_gen[s,:,t]
                
                plt.plot(X_gen_ssvec[0], X_gen_ssvec[2] )
                plt.plot(X_True_ssvec[0], X_True_ssvec[2])
                plt.show()
        return 
"""