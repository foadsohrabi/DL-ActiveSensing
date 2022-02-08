import numpy as np
import scipy.io as sio
from scipy.linalg import dft
import os

'This code is for generating channels according to the Rician Model. '
'Further, it generates the results for LMMSE benchmark and Perfect CSI benchmark'

def generate_channel(params_system, location_user_initial, Rician_factor,
                     scale_factor, num_samples, irs_Nh, is_Tx):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)
    location_irs = np.array([0, 0, 0])
    (num_elements_irs, num_user) = params_system
    channel_irs_user, set_location_user, channel_los = [], [], []
    i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
    i1 = np.reshape(i1,(1,num_elements_irs))
    i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)
    i2 = np.reshape(i2,(1,num_elements_irs))
    tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples,num_elements_irs, num_user]) \
          + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples,num_elements_irs, num_user])
    if is_Tx:
        phi = np.random.uniform(0,np.pi/2,size=[num_samples,1])
    else:
        phi = np.random.uniform(3*np.pi/2,2*np.pi,size=[num_samples,1])
    theta = np.random.uniform(-np.pi/2,np.pi/2,size=[num_samples,1])
    aoa_irs_y = np.cos(theta)*np.sin(phi)
    aoa_irs_z = np.sin(theta)
    a_irs_user = np.exp(1j * np.pi * ( aoa_irs_y*i1 +  aoa_irs_z*i2))
    alpha = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples,1]) \
          + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples,1])
    a_irs_user = alpha*a_irs_user
    channels = np.zeros([num_samples, num_elements_irs, num_user]) + 1j * np.zeros([num_elements_irs, num_user])
    channels[:, :,0] = np.sqrt(Rician_factor / (1 + Rician_factor)) * a_irs_user + np.sqrt(
        1 / (1 + Rician_factor)) * tmp[:,:,0]
    # channels[:, :,0] = a_irs_user
    return channels, set_location_user


def main_generate_channel(num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, location_user = None,
                          scale_factor = 120, isTesting = False):
    params_system = (num_elements_irs, num_user)
    if isTesting:
        if not os.path.exists('Rician_channel'):
            os.makedirs('Rician_channel')
        params_all = (num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor,scale_factor)
        file_test_channel = './Rician_channel/channel' + str(params_all) +'.mat'
        if os.path.exists(file_test_channel):
            data_test = sio.loadmat(file_test_channel)
            channel_tx, set_location_tx = data_test['channel_tx'], data_test['set_location_tx']
            channel_rx, set_location_rx = data_test['channel_rx'], data_test['set_location_rx']
        else:
            channel_tx, set_location_tx = generate_channel(params_system, location_user_initial=location_user,
                                                           Rician_factor=Rician_factor,
                                                           scale_factor=scale_factor, num_samples=num_samples,
                                                           irs_Nh=irs_Nh, is_Tx=True)
            channel_rx, set_location_rx = generate_channel(params_system, location_user_initial=location_user,
                                                           Rician_factor=Rician_factor,
                                                           scale_factor=scale_factor, num_samples=num_samples,
                                                           irs_Nh=irs_Nh, is_Tx=False)
            sio.savemat(file_test_channel,
                        {'channel_tx': channel_tx, 'set_location_tx': set_location_tx, 'channel_rx': channel_rx,
                         'set_location_rx': set_location_rx})
    else:
        channel_tx, set_location_tx = generate_channel(params_system, location_user_initial=location_user,
                                                       Rician_factor=Rician_factor,
                                                       scale_factor=scale_factor, num_samples=num_samples,
                                                       irs_Nh=irs_Nh, is_Tx=True)
        channel_rx, set_location_rx = generate_channel(params_system, location_user_initial=location_user,
                                                       Rician_factor=Rician_factor,
                                                       scale_factor=scale_factor, num_samples=num_samples,
                                                       irs_Nh=irs_Nh, is_Tx=False)
    return channel_tx, channel_rx


def generate_pilots(len_pilot, num_elements_irs, num_user):
    len_frame = num_user
    num_frame = len_pilot // len_frame
    if num_elements_irs>num_frame:
        phase_shifts = np.random.uniform(low=0,high=2*np.pi,size=(num_elements_irs, num_frame))
        phase_shifts = np.exp(1j*phase_shifts)
        phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)
    else:
        phase_shifts = dft(np.maximum(num_elements_irs,num_frame))
        phase_shifts = phase_shifts[0:num_elements_irs,0:num_frame]
        phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)
    pilots_subframe = dft(len_frame)
    pilots_subframe = pilots_subframe[:, 0:num_user]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = np.reshape(pilots, [len_pilot, num_user])
    return phase_shifts, pilots


def generate_received_pilots(channel_tx,channel_rx, phase_shifts, pilots, noise_power_db, scale_factor, Pt):
    [num_samples, num_elements_irs, num_user] = channel_tx.shape
    len_pilot = phase_shifts.shape[1]
    y = np.zeros((num_samples, num_user, len_pilot), dtype=complex)
    for k1 in range(num_user): # uplink receiver
        h_k1 = channel_tx[:,:,k1]
        for k2 in range(num_user): #uplink transmitter
            h_k2 = channel_rx[:,:,k2]
            h_cascaded = (h_k1*h_k2).reshape(-1,1,num_elements_irs)
            h_combined = h_cascaded@phase_shifts.reshape(1,num_elements_irs,len_pilot)
            pilots_k2 = pilots[:, k2]
            pilots_k2 = np.array([pilots_k2] * num_samples)
            pilots_k2 = pilots_k2.reshape((num_samples, 1, len_pilot))
            y[:,k1,:] = y[:,k1,:]+(h_combined*pilots_k2)[:,0,:]

    noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_user, len_pilot]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_user, len_pilot])
    noise_sqrt = np.sqrt(10 ** ((noise_power_db - Pt + scale_factor) / 10))
    y = y + noise_sqrt * noise
    y_real = np.concatenate([y.real, y.imag], axis=1)
    return y, np.array(y_real)


def decorrelation(received_pilots, pilots):
    (len_pilots, num_user) = pilots.shape
    (num_samples, num_user, _) = received_pilots.shape
    pilots = np.array([pilots] * num_samples)
    pilots = pilots.reshape((num_samples, len_pilots, num_user))

    len_frame = num_user
    num_frame = len_pilots // len_frame

    x_tmp = np.conjugate(pilots[:, 0:len_frame, :])
    y_decode = np.zeros([num_samples, num_user, num_user, num_frame], dtype=complex) #uplink: num_user_rx, num_user_tx
    for jj in range(num_frame):
        y_k = received_pilots[:, :, jj * len_frame:(jj + 1) * len_frame]
        y_decode_tmp = y_k @ x_tmp / len_frame
        y_decode[:, :, :, jj] = y_decode_tmp
    y_real = np.concatenate([y_decode.real, y_decode.imag], axis=-1)
    return y_decode, np.array(y_real)


def compute_stat_info(params_system, noise_power_db, phase_shifts, pilots, location_user, Rician_factor, irs_Nh, num_samples, scale_factor, Pt):
    (num_elements_irs, num_user) = params_system
    # phase_shifts, pilots = generate_pilots(len_pilot, num_elements_irs, num_user)
    channel_tx, set_location_tx = generate_channel(params_system, location_user_initial = location_user, Rician_factor = Rician_factor,
                                                    scale_factor=scale_factor, num_samples=num_samples, irs_Nh = irs_Nh, is_Tx = True)
    channel_rx, set_location_rx = generate_channel(params_system, location_user_initial = location_user, Rician_factor = Rician_factor,
                                                    scale_factor=scale_factor, num_samples=num_samples, irs_Nh = irs_Nh, is_Tx = False)
    y, y_real= generate_received_pilots(channel_tx, channel_rx, phase_shifts, pilots, noise_power_db, scale_factor, Pt)
    Y, Y_real = decorrelation(y, pilots)
    A = np.zeros((num_samples, num_user, num_elements_irs, num_user),dtype=complex)
    for kk in range(num_user):
        for jj in range(num_user):
            A[:,jj,:,kk] = channel_tx[:,:,jj]*channel_rx[:,:,kk]
    Q = phase_shifts[:,0:phase_shifts.shape[1]:num_user]
    A, Y = A[:, :, :, 0],  Y[:, :, 0, :]

    mean_A, mean_Y = np.mean(A, axis=0, keepdims=True), np.mean(Y, axis=0, keepdims=True)
    A = A - mean_A
    C_A = np.sum(np.matmul(np.transpose(A.conjugate(), (0, 2, 1)), A), axis=0) / num_samples
    Y = Y - mean_Y
    C_Y = np.sum(np.matmul(np.transpose(Y.conjugate(), (0, 2, 1)), Y), axis=0) / num_samples
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    stat_info = (gamma_n, C_A, mean_A)
    return stat_info


def ls_estimator(y, x):
    """
    y = h *x + n
    y: batch_size*m*l
    h: batch_size*m*n
    x: batch_size*n*l

    Output: h = y*x^H*(x*x^H)^-1
    """
    n, ell = x.shape[0], x.shape[1]
    x_H = np.transpose(x.conjugate())
    if ell < n:
        x_Hx = np.matmul(x_H, x)
        # print('Cond number:',np.linalg.cond(x_Hx))
        x_Hx_inv = np.linalg.inv(x_Hx)
        h = np.matmul(y, x_Hx_inv)
        h = np.matmul(h, x_H)
    elif ell == n:
        # print('Cond number:',np.linalg.cond(x))
        h = np.linalg.inv(x)
        h = np.matmul(y, h)
    else:
        xx_H = np.matmul(x, x_H)
        # print('Cond number:',np.linalg.cond(xx_H))
        xx_H_inv = np.linalg.inv(xx_H)
        h = np.matmul(y, x_H)
        h = np.matmul(h, xx_H_inv)
    return h


def lmmse_estimator(Y, Q, C_A, C_Y, mean_A, mean_Y):
    # # Y = AQ+N

    # ================================================
    # A = np.matmul(Y,np.linalg.inv(C_Y))
    # A = np.matmul(A,np.transpose(Q.conjugate()))
    # A = np.matmul(A,C_A)

    # ===============for numerical stability===========
    Y = Y - mean_Y
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    n, ell = Q.shape[0], Q.shape[1]
    if ell > n:
        QQ_H = np.matmul(Q, Q_H)
        C_A_inv = np.linalg.inv(C_A)
        tmp = np.linalg.inv(gamma_n * C_A_inv + QQ_H)
        tmp = np.matmul(tmp, QQ_H)
        tmp = np.matmul(C_A_inv, tmp)
        tmp = np.matmul(tmp, C_A)
        A = ls_estimator(Y, Q)
        A = np.matmul(A, tmp)
    else:
        tmp = np.matmul(Q_H, C_A)
        tmp = np.matmul(tmp, Q)
        tmp = tmp + gamma_n * np.eye(ell)
        tmp = np.linalg.inv(tmp)
        A = np.matmul(Y, tmp)
        A = np.matmul(A, Q_H)
        A = np.matmul(A, C_A)

    return A + mean_A


def channel_estimation_lmmse(params_system, y, pilots, phase_shifts, stat_info):
    (num_elements_irs, num_user) = params_system
    len_pilot = pilots.shape[0]
    num_sample = y.shape[0]
    len_frame = num_user
    Q = phase_shifts[:, 0:len_pilot:len_frame]
    (gamma_n, C_A, mean_A) = stat_info
    C_Y = np.matmul(np.matmul(np.transpose(Q.conjugate()), C_A), Q) + gamma_n * np.eye(Q.shape[1])
    mean_Y = np.matmul(mean_A, Q)
    y_d,_ = decorrelation(y, pilots)
    channel_cascaed = np.zeros((num_sample, num_user, num_elements_irs, num_user), dtype=complex)
    for kk in range(num_user):
        y_k = y_d[:, :, kk, :]
        channel_cascaed[:, :, :, kk]  = lmmse_estimator(y_k, Q, C_A, C_Y, mean_A, mean_Y)
    return channel_cascaed


def run_main_rate_pilot():
    num_ris_elements = 64
    num_user = 1
    params_system = (num_ris_elements, num_user)
    irs_Nh = 8
    Rician_factor = 10
    scale_factor = 120
    num_samples = 10000
    channel_tx_batch, channel_rx_batch = main_generate_channel(num_ris_elements, num_user, irs_Nh, num_samples,
                                                               Rician_factor, location_user=None, scale_factor=scale_factor,
                                                               isTesting=True)
    A = np.zeros((num_samples, num_user, num_ris_elements, num_user), dtype=complex)
    for kk in range(num_user):
        for jj in range(num_user):
            A[:, jj, :, kk] = channel_tx_batch[:, :, jj] * channel_rx_batch[:, :, kk]
    # ########### channel estimation
    noise_power_db = -scale_factor

    err_set = []
    optimal_power_set = []
    lmmse_power_set = []
    random_power_set = []
    len_pilot_set = num_user*np.array([4])
    P_u = 0
    for len_pilot in len_pilot_set:
        phase_shifts, pilots = generate_pilots(len_pilot, num_ris_elements, num_user)
        y,y_real = generate_received_pilots(channel_tx_batch, channel_rx_batch, phase_shifts, pilots, noise_power_db, scale_factor=scale_factor, Pt=P_u)
        stat_info = compute_stat_info(params_system, noise_power_db, phase_shifts, pilots,
                                      location_user=None, Rician_factor=Rician_factor,
                                      num_samples=10000, scale_factor=scale_factor, irs_Nh=irs_Nh, Pt=P_u)
        A_hat = channel_estimation_lmmse(params_system, y, pilots, phase_shifts, stat_info)
        err = np.linalg.norm(A-A_hat,ord='fro', axis=(1,2))**2/np.linalg.norm(A,ord='fro', axis=(1,2))**2
        err = np.mean(err,axis=0)
        err_set.append(err)

        h_input_val = np.squeeze(A)
        optimal_v_val = np.conj(h_input_val / np.abs(h_input_val))
        y_tmp_val = np.sum(np.multiply(optimal_v_val, h_input_val), axis=1)
        optimal_power_val = np.mean(np.abs(y_tmp_val) ** 2)
        optimal_power_set.append(10*np.log10(optimal_power_val))

        h_input_hat = np.squeeze(A_hat)
        optimal_v_val = np.conj(h_input_hat / np.abs(h_input_hat))
        y_tmp_val = np.sum(np.multiply(optimal_v_val, h_input_val), axis=1)
        lmmse_power_val = np.mean(np.abs(y_tmp_val) ** 2)
        lmmse_power_set.append(10*np.log10(lmmse_power_val))

        random_v = np.random.uniform(low=0, high=2 * np.pi, size=(num_samples, num_ris_elements))
        random_v = np.exp(1j * random_v)
        y_tmp_val = np.sum(np.multiply(random_v, h_input_val), axis=1)
        rand_power_val = np.mean(np.abs(y_tmp_val) ** 2)
        random_power_set.append(10*np.log10(rand_power_val))

        print('len_pilot', len_pilot, 'err:',err,'optimal_power',optimal_power_val,'lmmse_power',lmmse_power_val)

    print(optimal_power_set)
    print(lmmse_power_set)
    print(random_power_set)

    sio.savemat('lmmse_pm.mat',dict(lmmse_power_set=lmmse_power_set,
                                       pilots=len_pilot_set,N=num_ris_elements,
                                       optimal_power_set =optimal_power_set,
                                       random_power_set=random_power_set))


if __name__ == '__main__':
    run_main_rate_pilot()
