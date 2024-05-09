# -*- coding: utf-8 -*-
"""
Compute MSEs class
"""
import matplotlib.pyplot as plt
import time
from Plot import Plot
import numpy as np

class MSECalculator:
    def __init__(self):
        # Initializes any necessary constants or settings
        pass

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
                    resid_y = X_gen_ssvec[1] - X_True_ssvec[1]
                     
                    resid_squaresum += resid_x**2 + resid_y**2
             
                
                rmse = np.sqrt( resid_squaresum / X_True.shape[0]) 
                #totalrmse = residsquaresum
                square_errors_sum[t] = resid_squaresum
                mse_T[t] = rmse 
                #print('MSE-t[t]', mse_T)
            
            overall_rmse = np.sqrt((np.sum(square_errors_sum)) / (X_True.shape[2]*X_True.shape[0]))
            
            print('NEW SQRT OVERALL RMSE=', overall_rmse)
            
            return mse_T 
            
           
    def computeMSEKNet(
            self
            ) -> np.array:
        
        """
        computes MSE values for KNet method
        """
        
        return self.computeMSEsForSequences(self.test_target, self.KNet_est_out)
    
    
    def plotAllTraj(
        self,
        X_True: np.array,
        measurements: np.array,
        #X_gen_KNet: np.array,
        #X_gen_MLE: np.array,
        ) -> None:
        
        plt.figure()
        
        #print('xTrue2 shape', X_True.shape)
        for r in range(0, X_True.shape[0]):
            
            #print('X_True=', X_True)
            #print('x_gen_KNet shaoe', X_gen_KNet.shape)
            
            X_True_ssvec = X_True[r,:,:]
            measurements_ssvec = measurements[r,:,:]
         #   X_gen_KNet_ssvec = X_gen_KNet[r,:,:]
          #  X_gen_MLE_ssvec = X_gen_MLE[r,:,:]
            
            if r==14:
                plt.plot(X_True_ssvec[0], X_True_ssvec[1], color = 'k', label = 'ground truth')# label=f't={t}, s={s}')
          #      plt.plot(X_gen_KNet_ssvec[0], X_gen_KNet_ssvec[1], color = 'r', label = 'KalmanNet')
           #     plt.plot(X_gen_MLE_ssvec[0], X_gen_MLE_ssvec[1], color = 'g', label = 'Kalman-MLE')# label=f't={t}, s={s}')
                plt.plot(measurements_ssvec[0], measurements_ssvec[1], color = 'b', label = 'measurements', marker = 'x')
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