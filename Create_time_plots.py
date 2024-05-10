"""
Create time plots
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from compute_MSE_class import MSECalculator
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split

# Get the current working directory
#current_directory = os.getcwd()

folder_path = 'result_arrays'

dataFolderName = 'Simulations/Linear_canonical/H=I' + '/'
dataFileName =  ['2x2_rq2040_T100.pt', '2x2_rq-1010_T100.pt' ]     

[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName[0])
print('test_target', test_target.shape)

test_input_np = test_input.detach().cpu().numpy()
test_target_np = test_target.detach().cpu().numpy()

# Construct full paths
path_KF_state_est = os.path.join(folder_path, 'KF_state_est.pth')
path_KF_state_est_partial = os.path.join(folder_path, 'KF_state_est_partial.pth')
path_KNet_test_full = os.path.join(folder_path, 'KNet_test_full.pth')
#path_KNet_test_partial = os.path.join(folder_path, 'KNet_test_partial.pth')
#print('path_KF_state_est', path_KF_state_est)

# Load the files
KF_state_est = torch.load(path_KF_state_est, map_location=torch.device('cpu'))
KF_state_est_partial = torch.load(path_KF_state_est_partial, map_location=torch.device('cpu'))
KNet_test_full = torch.load(path_KNet_test_full, map_location=torch.device('cpu'))
#KNet_test_partial = torch.load(path_KNet_test_partial, map_location=torch.device('cpu'))

#print("Loaded 'KF_state_est':", type(KF_state_est))
#print("Loaded 'KF_state_est_partial':", type(KF_state_est_partial))
#print("Loaded 'KNet_test_full':", type(KNet_test_full))
#print("Loaded 'KNet_test_partial':", type(KNet_test_partial))

KF_state_est_np = KF_state_est.detach().numpy()
KF_state_est_partial_np = KF_state_est_partial.detach().numpy()
KNet_test_full_np = KNet_test_full.detach().numpy()
#KNet_test_partial_np = KNet_test_partial.detach().numpy()

#print('KF_state_est_np', KF_state_est_np.shape)
#print('KNET_test_full_np', KNet_test_full_np.shape)

mse_calculator = MSECalculator()

KF_est_mse = mse_calculator.computeMSEsForSequences(test_input_np, KF_state_est_np)
full_model_mse = mse_calculator.computeMSEsForSequences(test_input_np, KNet_test_full_np)





#partial_model_mse = mse_calculator.computeMSEsForSequences(KF_state_est_partial_np, KNet_test_partial_np)
#print('full_model_mse', full_model_mse.shape)

#print('KF_state_est_np', KF_state_est_np)
#print('KF_state_est_np shape', KF_state_est_np.shape)


#plt.plot(KF_state_est_np[0], KF_state_est_np[1])
#plt.plot(KF_state_est_partial_np[0], KF_state_est_partial_np[1])

print('test_TARGETTTTT target', test_target_np[0].shape)
mse_calculator.plotAllTraj(KF_state_est_np, KNet_test_full_np, test_target_np)

plt.xlim(1,100) #100)
plt.plot(np.arange(0,100,1), full_model_mse, label = 'KNET')
plt.plot(np.arange(0,100,1), KF_est_mse, label = 'KF' )
plt.legend()
plt.title('full model info, 500 epochs, r-1e-2')#
plt.xlabel('k')
plt.ylabel('RMSE')
plt.show()


#plot partial model
#plt.xlim(1,100)
#plt.plot(np.arange(0,100,1), partial_model_mse)
#plt.title('full model info, 50 epochs')#
#plt.xlabel('k')
#plt.ylabel('RMSE')
#plt.show()

"""
dataFolderName = 'Simulations/Linear_canonical/H=I' + '/'
dataFileName = ['2x2_rq-1010_T100.pt','2x2_rq020_T100.pt','2x2_rq1030_T100.pt','2x2_rq2040_T100.pt','2x2_rq3050_T100.pt']

DatafolderName = 'Filters/Linear' + '/'
DataResultName = 'KF_'+ dataFileName[0]
     
fullPath = DatafolderName + DataResultName
#print("Full path to the file:", fullPath)


data = torch.load(fullPath)
#print('data', data)

x = np.arange(0,200, 1)
y = data['MSE_KF_linear_arr'].cpu().numpy()

#print('y', type(y))

plt.plot(x,y)

Path_to_results = 'KNet' +'/'
DFN = 'pipeline_KNet_2x2_rq-1010_T100.pt'

KNet_test_full = torch.load(Path_to_results + DFN)
print('KNet_test_fullModel', dir(KNet_test_full))
"""