# -*- coding: utf-8 -*-
"""
Run Monte Carloes
"""
import MC_sim 
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

n_steps = 20
resid_squaresum = np.zeros(n_steps)

startNNTrainTS = pd.Timestamp.utcnow()

print("training mcs net")
mcs_train = MC_sim.MonteCarloSimulation(n_steps,314) 
mcs_train.generateTrajTrainKNet()
endNNTrainTS = pd.Timestamp.utcnow()
print("training mcs net done, time taken =", endNNTrainTS - startNNTrainTS)
mcs_train.loadTrajTestKNet()
mcs_train.applyKalmanFilterKNet()


startMLTrainTS = pd.Timestamp.utcnow()
print("generating mcs estimate of Q and R")
mcs_est = MC_sim.MonteCarloSimulation(n_steps,314)
mcs_est.generateTrajectory()
mcs_est.generateEstimatesMaxLk()
endMLTrainTS = pd.Timestamp.utcnow()
print("generating mcs estimate of Q and R done, time taken=", endMLTrainTS - startMLTrainTS)
mcs_est.applyKalmanFilterMaxLk()

MSE_maxLK = mcs_est.computeMSEMaxLk()
MSE_KNet = mcs_train.computeMSEKNet()

average_maxLK = np.mean(MSE_maxLK)
average_KNet = np.mean(MSE_KNet)
#average_MSE_maxLK = mcs_est.computeAverageMSEMLE()
#average_MSE_KNet = mcs_est.computeAverageMSEKNet()


#create plot
xarr = np.arange(n_steps)
plt.plot(xarr, MSE_maxLK, color = 'blue', label = 'MLE')
plt.plot(xarr, MSE_KNet,color = 'red', label = 'Kalman net')
plt.title("KF MSE comparison with NN vs MLE learned parameters ")
plt.xlabel('Time Step')
plt.ylabel('Average MSE')
plt.legend()
plt.grid()
plt.savefig('MSE_plot_KN')
plt.show()
 
    
 