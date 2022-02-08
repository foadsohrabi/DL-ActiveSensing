import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["New Century Schoolbook"],
})


def compute_prob_posterior(y, P_snr, W, A_dic, prob_prior):
    mean_all = complex(np.sqrt(P_snr)) * np.transpose(np.conj(W), axes=(0, 2, 1)) @ A_dic
    dist = np.sum(np.abs(y - mean_all) ** 2, axis=1)
    dist = dist - np.min(dist, axis=-1, keepdims=True)
    prob_likely = np.exp(-dist)
    prob_posterior = prob_prior * prob_likely
    prob_posterior = prob_posterior / np.sum(prob_posterior, axis=1, keepdims=True)
    return prob_posterior


def estimate_AoA(prob_aoa, phi_all, phi_true):
    # MAP
    lamda_t = np.argmax(prob_aoa, axis=1)
    phi_hat = phi_all[lamda_t]
    mse_phi_MAP = np.mean((np.squeeze(phi_hat) - np.squeeze(phi_true)) ** 2)
    # MMSE
    num_AoAs = phi_all.shape[0]
    phi_hat_mmse = np.reshape(prob_aoa, [-1, 1, num_AoAs]) @ np.reshape(phi_all, [1, num_AoAs, 1])
    mse_phi_mmse = np.mean((np.squeeze(phi_hat_mmse) - np.squeeze(phi_true)) ** 2)
    return [mse_phi_MAP, mse_phi_mmse, np.squeeze(phi_hat_mmse)]


'load data'
data = sio.loadmat('array_respons_LSTM_tau12.mat')
phi_val = data['phi_true'].squeeze()
W_her = data['W_all']
W = np.transpose(np.conj(W_her), axes=(0, 2, 1))
y = data['y_all']
phi_est = data['phi_est'].squeeze()
err = (phi_est - phi_val) ** 2
batch_size_test = W_her.shape[0]

'Problem setup'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
snrdB = 0.0  # SNR in dB
P_snr = 10 ** (snrdB / 10)  # Considered TX powers
N = 64

'Discretize AoA to calculate posterior probability'
num_AoAs = 10000
phi_all = np.linspace(start=phi_min, stop=phi_max, num=num_AoAs)
phi_all_degree = phi_all * 180 / np.pi
A_dic = np.exp(1j * np.pi * np.reshape(np.arange(N), (N, 1)) * np.sin(phi_all))

'Beamforming gain'
bf_gain = 10 * np.log10(np.abs(W_her @ A_dic) ** 2)

'Plot figures'
plot_idx = 1
tau = 8
fig1, axs1 = plt.subplots(tau, figsize=(5, 10))
# fig1.suptitle('Array response (True AoA=%.2f)'
#               % (phi_val[plot_idx] * 180 / np.pi), fontsize=20)
fig1.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('AoA in Degree (True AoA=$%.2f^\circ$)' % (phi_val[plot_idx] * 180 / np.pi), fontsize=15)
# plt.xlabel('AoA in degree (True AoA=$9.96^\circ$)', fontsize=15)
plt.ylabel('Beamforming Gain (dB)', fontsize=15, labelpad=20)

fig2, axs2 = plt.subplots(tau, figsize=(5, 10))
# fig2.suptitle('Posterior probability (True AoA=%.2f)'
#               % (phi_val[plot_idx] * 180 / np.pi), fontsize=20)
fig2.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('AoA in Degree (True AoA=$%.2f^\circ$)' % (phi_val[plot_idx] * 180 / np.pi), fontsize=15)
# plt.xlabel('AoA in degree (True AoA=$9.96^\circ$)', fontsize=15)
plt.ylabel('Posterior Distribution of AoA', labelpad=20, fontsize=15)
prob_initial = np.ones([batch_size_test, num_AoAs]) / num_AoAs
prob_pos = prob_initial
axs2[0].plot(phi_all_degree, prob_pos[plot_idx, :], lw=2)
axs2[0].set_title('Sensing Time Frame t=%d' % 0)
diff = []
for ii in range(tau):
    axs1[ii].plot(phi_all_degree, bf_gain[plot_idx, ii, :], lw=2)
    axs1[ii].set_title('Sensing Time Frame t=%d' % (ii + 1))
    axs1[ii].set_ylim(bottom=-30, top=15)
    # prob_pos1 = compute_prob_posterior(y[:,0:ii+1,:], P_snr, W[:,:,0:ii+1], A_dic, prob_initial)
    # axs1[ii].plot(phi_all_degree, prob_pos1[plot_idx,:])
    prob_pos = compute_prob_posterior(y[:, ii:ii + 1, :], P_snr, W[:, :, ii:ii + 1], A_dic, prob_pos)
    # diff.append(prob_pos-prob_pos1)
    if ii < tau - 1:
        axs2[ii + 1].plot(phi_all_degree, prob_pos[plot_idx, :], lw=2)
        axs2[ii + 1].set_title('Sensing Time Frame t=%d' % (ii + 1))
    # axs2[ii + 1].plot(phi_all_degree, prob_pos[plot_idx, :], lw=2)
    # axs2[ii + 1].set_title('Sensing Time Frame T=%d' % (ii + 1))

# tmp = estimate_AoA(prob_pos, phi_all, phi_val)
fig1.tight_layout()
fig2.tight_layout()

plt.show()
fig1.savefig('./beam_pattern/figures/bf_lstm_tau12.pdf', bbox_inches='tight', pad_inches=0.05)
fig2.savefig('./beam_pattern/figures/pos_lstm_tau12.pdf', bbox_inches='tight', pad_inches=0.05)
