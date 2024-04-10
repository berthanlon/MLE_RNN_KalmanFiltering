# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:52:39 2023

@author: sgbhanlo
"""
######## RUN ON CPU VERSION 
import torch
import config 
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)

import MC_sim_2 
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd 
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('mode')
#args = parser.parse_args()

####################################
### Generative Parameters For CA ###
####################################
args = config.general_settings()

### Dataset parameters
args.N_E = 10
args.N_CV = 4
args.N_T = 2
#offset = 0 ### Init condition of dataset
args.randomInit_train = False
args.randomInit_cv = False
args.randomInit_test = False

args.Tlen = 5
args.Tlen_test = 5

### training parameters
#KnownRandInit_train = True # if true: use known random init for training, else: model is agnostic to random init
#KnownRandInit_cv = True
#KnownRandInit_test = True
args.use_cuda = True # use GPU or not
args.nSteps = 20
args.n_batch = 10
args.lr = 1e-4
args.wd = 1e-4

#n_steps = 10

mode = 'datagen' # args.mode
print(f"mode = '{mode}'")
base_dir = f'C:/Users/betti/Desktop/MLE_KNET_architecture_2/KNetFiles_{args.nSteps}/'
fname_base = 'MCSim_test'
###resid_squaresum = np.zeros(n_steps)

startTS = pd.Timestamp.utcnow()

fname_data = fname_base + f'_data_{args.nSteps}'
fname_mse_KNet = base_dir + fname_base + f'_KNet_{args.nSteps}'
fname_mse_MLE = base_dir + fname_base + f'_MLE_{args.nSteps}'
fname_plot = base_dir + fname_base + f'_plot_{args.nSteps}'

if mode == 'datagen':        
    print("training mcs net")
    mcs_train = MC_sim_2.MonteCarloSimulation(args.nSteps, base_dir, fname_data, args, 311) 
    mcs_train.generateTrajTrainKNet()
    endTS = pd.Timestamp.utcnow()
    print("training mcs net done, time taken =", endTS - startTS)

elif mode == 'knet':
    print("computing mses for knet")
    mcs_train = MC_sim_2.MonteCarloSimulation(args.nSteps, base_dir, fname_data, 312)
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
    mcs_est = MC_sim_2.MonteCarloSimulation(args.nSteps, base_dir, fname_data, 313)
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
    mcs_plt = MC_sim_2.MonteCarloSimulation(args.nSteps, base_dir, fname_data, 313) 
    mcs_plt.runPlotALL()
    
    MSE_KNet_torch = torch.load(fname_mse_KNet)
    MSE_KNet = MSE_KNet_torch.cpu().numpy()
    
    MSE_maxLK_torch = torch.load(fname_mse_MLE)
    MSE_maxLK = MSE_maxLK_torch.cpu().numpy()
    
    xarr = np.arange(args.nSteps)
    plt.plot(xarr, MSE_KNet, color = 'red', label = 'KalmanNet')
    plt.plot(xarr, MSE_maxLK, color = 'blue', label = 'Kalman-MLE')
    plt.xlim(0,args.nSteps-1)
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



