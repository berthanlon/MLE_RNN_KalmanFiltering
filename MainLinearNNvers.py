import numpy as np
import matplotlib.pyplot as plt
from pellipse import pellipse
import torch
import torch.nn as nn
import torch.nn.functional as func
from Linear_sysmdl import SystemModel

dev = torch.device("cpu")

np.random.seed(12)


mean_ini = np.array([0, 2, 0, 2])
P_ini = np.diag([1, 0.1, 1, 0.1])
chol_ini = np.linalg.cholesky(P_ini)

# Nearly constant velocity model
T = 0.5 # sampling time
T_test = T


F = torch.tensor([[1, T, 0, 0], 
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]]) # state transition matrix
sigma_u = 0.5 # standard deviation of the acceleration
Q = torch.tensor([[T**3/3, T**2/2, 0, 0], 
              [T**2/2, T, 0, 0],
              [0, 0, T**3/3, T**2/2],
              [0, 0, T**2/2, T]]) * sigma_u**2 # covariance of the process noise


#Q=np.eye(4)
m = 2

chol_Q = np.linalg.cholesky(Q) # Cholesky decomposition of Q

# Measurement model
H = torch.tensor([[1, 0, 0, 0],
              [0, 0, 1, 0]]) # measurement matrix
R = torch.tensor(0.5 * np.diag([1, 1])) # covariance of the measurement noise
chol_R = np.linalg.cholesky(R) # Cholesky decomposition of R

m1_0 = torch.tensor([[0.0], [0.0]]).to(dev)
# m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2_0 = 0 * 0 * torch.eye(m).to(dev)


# Number of time steps
n_steps = 20000


# Generate ground truth
X_true = np.zeros((4, n_steps)) # state vector (x, x_vel, y, y_vel)
X_true[:, 0] = mean_ini + np.dot(chol_ini, np.random.randn(4))


# Generate measurements
measurements = np.zeros((2, n_steps))
measurements[:, 0] = np.dot(H, X_true[:, 0]) + np.dot(chol_R, np.random.randn(2))
#print(measurements.shape)

# Generate the complete trajectory and measurements
for k in range(1, n_steps):
    # Propagate the true state
    X_true[:, k] = np.dot(F, X_true[:, k-1]) + np.dot(chol_Q, np.random.randn(4))
    # Generate measurement
    measurements[:, k] = np.dot(H, X_true[:, k]) + np.dot(chol_R, np.random.randn(2))

#initialise true traj
#meas_diff = np.zeros((2, n_steps-1))
#meas_resi = np.zeros((1, n_steps-1))

#X_diff_Q_sum = np.array(np.zeros((4,4)))
#R_sum = np.array(np.zeros((2,2)))
#print('X_diff_Q_sum', X_diff_Q_sum)


# Generate true trajectory
#####call in NN here?

print('creating sys model')
sys_model = SystemModel(F, Q, H, R, T, T_test)
sys_model.InitSequence(m1_0, m2_0)
 

print('creating sys model done')








# Apply the Kalman filter
x_k = mean_ini # initial state estimate
P_k = P_ini # initial covariance estimate

kalman_est = np.zeros((4, n_steps)) # state estimate
error_ellipse = [] # error ellipse

for k in range(0, n_steps):
    # Update step
    K_gain = P_k @ H.T @ np.linalg.inv( (H @ P_k @ H.T) + R_est)
    x_u = x_k + K_gain @ (measurements[:, k] - H @ x_k) # updated state estimate
    P_u = (np.eye(4) - K_gain @ H) @ P_k # updated covariance estimate

    kalman_est[:, k] = x_u # Filtered estimate kalman_est = x_u
    
    # kalman_est[:,k] = F @ x_u # for the measurement prediction i.e kalman_est = H(Fx_u)

    # Predict step
    x_k = F @x_u# predicted state estimate
    P_k = F @ P_u @ F.T + Q_est # predicted covariance estimate

    error_ellipse.append(pellipse(x_k[[0,2]], P_k[[0,2],:][:,[0,2]], 100, 4)) # H(x_k), H(P_u)H.T  ->   H(P_u)H.T + R ? 


# Plot the results
plt.plot(X_true[0, :], X_true[2, :], 'b', marker='o', label='True trajectory')
plt.plot(measurements[0, :], measurements[1, :], 'k+', label='Measurements')
plt.plot(kalman_est[0, :], kalman_est[2, :], 'r', marker='o', label='Kalman filter estimate')
for i in range(n_steps):
    plt.plot(error_ellipse[i][0], error_ellipse[i][1], 'g', linewidth=1, label='Error ellipse' if i==0 else None)
ax = plt.gca()
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.title('Kalman filter demo')
plt.savefig('Bettinacodegraph200ts')
plt.show()
