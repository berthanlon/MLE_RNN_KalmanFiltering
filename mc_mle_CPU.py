# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:52:39 2023

@author: sgbhanlo
"""
######## RUN ON CPU VERSION 
import torch
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)

import MC_sim 
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('mode')
#args = parser.parse_args()

n_steps = 100

mode = 'datagen' # args.mode
print(f"mode = '{mode}'")
base_dir = f'C:/Users/betti/Desktop/MLE_KNET_Range_bearing_a1/KNetFiles_{n_steps}/'
fname_base = 'MCSim_test'

###resid_squaresum = np.zeros(n_steps)

startTS = pd.Timestamp.utcnow()

fname_data = fname_base + f'_data_{n_steps}'
fname_mse_KNet = base_dir + fname_base + f'_KNet_{n_steps}'
fname_mse_MLE = base_dir + fname_base + f'_MLE_{n_steps}'
fname_plot = base_dir + fname_base + f'_plot_{n_steps}'

if mode == 'datagen':        
    print("training mcs net")
    mcs_train = MC_sim.MonteCarloSimulation(n_steps, base_dir, fname_data, 311) 
    mcs_train.generateTrajTrainKNet()
    endTS = pd.Timestamp.utcnow()
    print("training mcs net done, time taken =", endTS - startTS)


elif mode == 'knet':
    print("computing mses for knet")
    mcs_train = MC_sim.MonteCarloSimulation(n_steps, base_dir, fname_data, 312)
    mcs_train.loadTrajTestKNet()
    mcs_train.applyKalmanFilterKNet()
    MSE_KNet = mcs_train.computeMSEKNet()
    MSE_KNet_torch = torch.tensor(MSE_KNet)
    torch.save(MSE_KNet_torch, fname_mse_KNet)
    
    mcs_train.allTrajKNet()
    
    #average_KNet = np.mean(MSE_KNet)
    endTS = pd.Timestamp.utcnow()
    print(f"computing mses for knet done, time taken = {endTS - startTS}")

elif mode == 'mle':
    print("generating mcs estimate of Q and R")
    mcs_est = MC_sim.MonteCarloSimulation(n_steps, base_dir, fname_data, 313)
    mcs_est.generateEstimatesMaxLk() 
    mcs_est.applyKalmanFilterMaxLk()
    MSE_maxLK = mcs_est.computeMSEMaxLk()
    MSE_maxLK_torch = torch.tensor(MSE_maxLK)
    torch.save(MSE_maxLK_torch, fname_mse_MLE)
    
    mcs_est.allTrajMLE()
    
    #average_maxLK = np.mean(MSE_maxLK)
    endTS = pd.Timestamp.utcnow()
    print(f"generating mcs estimate of Q and R done, time taken = {endTS - startTS}")
    
elif mode == 'plot':
    print("generating plots")
    mcs_plt = MC_sim.MonteCarloSimulation(n_steps, base_dir, fname_data, 313) 
    mcs_plt.runPlotALL()
    
    MSE_KNet_torch = torch.load(fname_mse_KNet)
    MSE_KNet = MSE_KNet_torch.cpu().numpy()
    
    MSE_maxLK_torch = torch.load(fname_mse_MLE)
    MSE_maxLK = MSE_maxLK_torch.cpu().numpy()
    
    xarr = np.arange(n_steps)
    plt.plot(xarr, MSE_KNet, color = 'red', label = 'KalmanNet')
    plt.plot(xarr, MSE_maxLK, color = 'blue', label = 'Kalman-MLE')
    plt.xlim(0,n_steps-1)
    plt.ylim(0,None)
    #plt.title("KF MSE comparison with NN vs MLE learned parameters ")
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    #plt.savefig('fname_plot')
    plt.savefig(f'{fname_plot}.eps',format="eps")
    plt.show()
    print("generating plots done")



