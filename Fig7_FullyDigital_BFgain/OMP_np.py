import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
np.random.seed(1)

'This code generate the results for MRT w$/$ perfect CSI MRT w/ OMP channel estimation (random sensing, fixed)'

def func_codedesign(delta_inv, phi_min, phi_max, N1):
    phi = np.linspace(start=phi_min, stop=phi_max, num=delta_inv)
    A_tmp = np.zeros([N1, delta_inv], dtype=np.complex128)
    for i in range(delta_inv):
        from0toN1 = np.float32(list(range(0, N1)))
        a_phi1 = np.exp(1j * np.pi * from0toN1 * np.sin(phi[i]))
        A_tmp[:,i] = a_phi1
    return A_tmp, phi

'System Information'
N1 = 64  # Number of BS's antennas
tau1 = 20 # Pilot length
L = 2    # number of path

'Channel Information'
phi_min = -60*(np.pi/180) #Lower-bound of AoAs
phi_max = 60*(np.pi/180) #Upper-bound of AoAs
num_SNR = 7 #Number of considered SNRs
idx_SNR_val = 0 #Index of SNR for validation (saving parameters)
snrdBvec = np.array([25])
Pvec = 10**(snrdBvec/10) #Set of considered TX powers
mean_true_alpha = 0.0 + 0.0j #Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5) #STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5) #STD of the Gaussian noise per real dim.
delta_inv = 512

A_dic, phi_all = func_codedesign(delta_inv, phi_min, phi_max, N1)
batch_size_test = 10000 #Validation_set_size = val_size_order*delta_inv
###############  Validation Set  #######################
alpha_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_test,L])\
            +1j*np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_test,L])
phi_1_idx_val = np.ones((batch_size_test,L))
for ii in range(batch_size_test):
    phi_1_idx_val[ii] = np.random.choice(delta_inv,replace=False ,size=[L])
phi_1_idx_val=np.sort(phi_1_idx_val,axis=-1)
phi_1_val = phi_all[phi_1_idx_val.astype(int)]

##############Multi-path channel construction#######################
from0toN1 = np.float32(list(range(0, N1)))
a_phi1 = 0
for ii in range(L):
    a_phi1 = a_phi1+ np.reshape(alpha_val[:, ii], [-1, 1, 1])*np.exp(1j * np.pi * from0toN1.reshape(-1,N1,1) * np.sin(phi_1_val[:,ii]).reshape(-1,1,1))
a_phi1 = a_phi1/np.sqrt(L)
a_phi1 = np.reshape(a_phi1,(-1,N1,1))
v_her_opt = a_phi1
v_her_opt = v_her_opt / np.linalg.norm(v_her_opt, axis=1, keepdims=True)
tmp_opt = np.transpose(np.conj(v_her_opt), axes=[0, 2, 1]) @ a_phi1
bf_gain_opt = np.mean(np.abs(tmp_opt) ** 2)
##################random beamforming matrix###################################
V = np.random.normal(loc=0, scale=1, size=[batch_size_test,tau1,N1])\
    +1j*np.random.normal(loc=0, scale=1, size=[batch_size_test,tau1,N1])
V = V/np.linalg.norm(V,axis=2,keepdims=True)

v_her_rnd = np.random.normal(loc=0, scale=1, size=[batch_size_test,N1,1])\
    +1j*np.random.normal(loc=0, scale=1, size=[batch_size_test,N1,1])
v_her_rnd = v_her_rnd / np.linalg.norm(v_her_rnd, axis=1, keepdims=True)
tmp_opt = np.transpose(np.conj(v_her_rnd), axes=[0, 2, 1]) @ a_phi1
bf_gain_random = np.mean(np.abs(tmp_opt) ** 2)
####################channel model######################################
P = Pvec[idx_SNR_val]
Y_noiseless =V@a_phi1
noise = np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(Y_noiseless))\
    +1j*np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(Y_noiseless))
y_vec = np.complex(np.sqrt(P))*Y_noiseless+noise
#########OMP algorithm###########################################
mse_aoa = []
mse_loss = []
bf_gain_omp_list = []
for ii in range(batch_size_test):
    A = np.complex(np.sqrt(P))*V[ii]@A_dic
    idx = np.zeros(L,dtype=int)
    r = y_vec[ii]
    for tt in range(L):
        lamda_t = np.argmax(np.abs(np.transpose(np.conj(r))@A))
        idx[tt] = lamda_t
        phi_A = np.reshape(A[:, idx[0:tt+1]], (-1, tt+1))
        alpha_hat = np.linalg.inv(np.transpose(np.conj(phi_A)) @ phi_A) @ np.transpose(np.conj(phi_A)) @ y_vec[ii]
        ### if np.transpose(np.conj(phi_A)) @ phi_A is not invertible, please use the code in line 87 to compute alpha_hat
        # alpha_hat = np.transpose(np.conj(phi_A)) @ np.linalg.inv(phi_A@np.transpose(np.conj(phi_A))) @  y_vec[ii]
        r = y_vec[ii]-phi_A@alpha_hat
    alpha_hat = alpha_hat*np.sqrt(L)
    a_phi1_hat = A_dic[:,idx]@alpha_hat/np.sqrt(L)
    mse_loss.append(np.linalg.norm(a_phi1_hat-a_phi1[ii])**2)
    aoa_true = np.sort(phi_1_val[ii])
    aoa_hat = np.sort(phi_all[idx])
    mse_aoa.append(np.sum((aoa_hat-aoa_true)**2))

    v_hat = a_phi1_hat / np.linalg.norm(a_phi1_hat)
    bf_gain_omp_list.append(np.mean(np.abs(np.transpose(np.conj(v_hat)) @ a_phi1[ii]) ** 2))
    bf_gain_omp = np.mean(bf_gain_omp_list)
    if ii%10==0:
        print('ii:%3d'%ii, ' SNR:',snrdBvec[idx_SNR_val], 'mse_loss:%2.5f'%(np.mean(mse_aoa)),'  bf_gain_omp:%2.5f'%bf_gain_omp, '  bf_gain_opt:%2.5f'%bf_gain_opt)

mse_aoa = np.mean(mse_aoa)
bf_gain_omp = np.mean(bf_gain_omp_list)
print('SNR:',snrdBvec[idx_SNR_val], 'mse_loss:',mse_aoa, '  bf_gain_omp:',bf_gain_omp,'  bf_gain_opt:',bf_gain_opt,'   random_bf:',bf_gain_random)
