# -*- coding: utf-8 -*-
"""
Run Monte Carloes
"""
import MC_sim 
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

n_sims = 2

n_steps = 7
resid_squaresum = np.zeros(n_steps)

startNNTrainTS = pd.Timestamp.utcnow()

print("training mcs net")
mcs_train = MC_sim.MonteCarloSimulation(n_steps, 314) #
#mcs_train.generateTrajTrainKNet()
endNNTrainTS = pd.Timestamp.utcnow()
print("training mcs net done, time taken =", endNNTrainTS - startNNTrainTS)
#mcs_train.loadTrajTestKNet()
mcs_train.applyKalmanFilterKNet()


startMLTrainTS = pd.Timestamp.utcnow()
print("generating mcs estimate of Q and R")
mcs_est = MC_sim.MonteCarloSimulation(n_steps)
#mcs_est.generateTrajectory()
mcs_est.generateEstimatesMaxLk()
endMLTrainTS = pd.Timestamp.utcnow()
print("generating mcs estimate of Q and R done, time taken=", endMLTrainTS - startMLTrainTS)
mcs_est.applyKalmanFilterMaxLk()


#for i in range(1, n_sims + 1 ):
#    print(f"Running MC number:\n{i}")
#    
#    mcs = MC_sim.MonteCarloSimulation(n_steps) #


#   mcs.applyKalmanFilterMaxLk()

    
 
    
 
    
 
 
"""
 X_true_pos = trajArrays["X_true"][[0, 2], :]
 measurements = trajArrays["measurements"]    
 
 X_true_pos_tensor = torch.from_numpy(X_true_pos)
 measurements_tensor = torch.from_numpy(measurements)
 print("X_True_x_tensor, measurements_tensor", X_true_pos_tensor, measurements_tensor)
 
 #print("X_True_x, X_True_y", X_True_x, X_True_y)
 
 #print(f"x_k = \n{mcs.x_k}")
 #print(f"x_k_o) = \n{mcs.x_k_o}")
 

# print(f"Kest: \n{mcs.kalman_est}")
# print(f"X_true: \n{mcs.X_true}")
 
 resid_x = mcs.kalman_est[0] - mcs.X_true[0]
 resid_y = mcs.kalman_est[2] - mcs.X_true[2]
 
 resid_squaresum += resid_x**2 + resid_y**2
 
rmse = np.sqrt( resid_squaresum / n_sims  ) 
    
#print(f"rmse:\n{rmse}")
plt.plot(range(2, n_steps+1), rmse[1:n_steps])

    #print(f"xdiff: \n{xdiff}")
 #   print(f"rmse: \n{rmse}")
 
     """