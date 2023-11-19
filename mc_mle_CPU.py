# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:52:39 2023

@author: sgbhanlo
"""
######## RUN ON CPU VERSION 


import MC_sim 
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('mode')
#args = parser.parse_args()

mode = 'plot' # args.mode
print(f"mode = '{mode}'")
base_dir = 'C:/Users/sgbhanlo/Documents/KalmanNet_TSP-main/KNetFiles/'
fname_base = 'MCSim_test'
n_steps = 50
###resid_squaresum = np.zeros(n_steps)

startTS = pd.Timestamp.utcnow()

fname_data = fname_base + '_data50'
fname_mse_KNet = base_dir + fname_base + '_KNet'
fname_mse_MLE = base_dir + fname_base + '_MLE'
fname_plot = base_dir + fname_base + '_plot'

if mode == 'datagen':        
    print("training mcs net")
    mcs_train = MC_sim.MonteCarloSimulation(n_steps, fname_data, 314) 
    mcs_train.generateTrajTrainKNet()
    endTS = pd.Timestamp.utcnow()
    print("training mcs net done, time taken =", endTS - startTS)

elif mode == 'knet':
    print("computing mses for knet")
    mcs_train = MC_sim.MonteCarloSimulation(n_steps, fname_data, 314)
    mcs_train.loadTrajTestKNet()
    mcs_train.applyKalmanFilterKNet()
    MSE_KNet = mcs_train.computeMSEKNet()
    MSE_KNet_torch = torch.tensor(MSE_KNet)
    torch.save(MSE_KNet_torch, fname_mse_KNet)
    average_KNet = np.mean(MSE_KNet)
    endTS = pd.Timestamp.utcnow()
    print(f"computing mses for knet done, time taken = {endTS - startTS}, average_KNet = {average_KNet}")

elif mode == 'mle':
    print("generating mcs estimate of Q and R")
    mcs_est = MC_sim.MonteCarloSimulation(n_steps, fname_data, 314)
#    mcs_est.generateTrajectory()
    mcs_est.generateEstimatesMaxLk()
    mcs_est.applyKalmanFilterMaxLk()
    MSE_maxLK = mcs_est.computeMSEMaxLk()
    MSE_maxLK_torch = torch.tensor(MSE_maxLK)
    torch.save(MSE_maxLK_torch, fname_mse_MLE)
    average_maxLK = np.mean(MSE_maxLK)
    endTS = pd.Timestamp.utcnow()
    print(f"generating mcs estimate of Q and R done, time taken = {endTS - startTS}, average_maxLK = {average_maxLK}")
    
elif mode == 'plot':
    print("generating plots")
    MSE_KNet_torch = torch.load(fname_mse_KNet)
    MSE_KNet = MSE_KNet_torch.numpy()
    
    MSE_maxLK_torch = torch.load(fname_mse_MLE)
    MSE_maxLK = MSE_maxLK_torch.numpy()
    
    xarr = np.arange(n_steps)
    plt.plot(xarr, MSE_KNet, color = 'red', label = 'Kalman net')
    plt.plot(xarr, MSE_maxLK, color = 'blue', label = 'MLE')
    plt.title("KF MSE comparison with NN vs MLE learned parameters ")
    plt.xlabel('Time Step')
    plt.ylabel('Average MSE')
    plt.legend()
    plt.grid()
    plt.savefig(fname_plot)
    plt.show()
    print("generating plots done")
 