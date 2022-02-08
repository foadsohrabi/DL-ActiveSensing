import numpy as np


def compute_weighted_sum_rate2(channel_tx, channel_rx, v, p, alpha, sigma2_dB):
    '''
    :param channel_tx: [num_samples, num_elements_irs, num_user]
    :param channel_rx: [num_samples, num_elements_irs, num_user]
    :param alpha: [num_samples, num_user] weights of rate
    :param v: [num_samples, num_elements_irs, 1]
    :param p: [num_samples, num_user]
    :param sigma2_dB: scalar in dB
    :param channel_cascaded: [num_samples,num_user_tx,num_elements_irs, num_user_rx]
    :return: average sum rate of the dataset
    '''
    sigma_sqr = 10**(sigma2_dB/10)
    # sigma_sqr = 0
    [num_samples, num_elements_irs, num_user] = channel_tx.shape
    v = np.reshape(v,[-1,num_elements_irs, 1])
    p = np.reshape(p,[-1, num_user])
    rate_set = []
    weighted_rate_set = []
    direct_power_set = []
    interference_set = []
    for kk in range(num_user):  # rx
        ch_kk = np.reshape(channel_tx[:, :, kk] * channel_rx[:, :, kk], [-1, 1, num_elements_irs])
        sinr_num_kk = p[:, kk] * np.squeeze((np.abs(ch_kk @ v) ** 2))
        direct_power_set.append(sinr_num_kk)
        sinr_de_kk = sigma_sqr  # noise power
        for jj in range(num_user):
            if jj != kk:
                ch_kj = np.reshape(channel_tx[:, :, jj] * channel_rx[:, :, kk], [-1, 1, num_elements_irs])
                sinr_de_kk = sinr_de_kk + p[:, jj] * np.squeeze((np.abs(ch_kj @ v) ** 2))
        interference_set.append(sinr_de_kk - sigma_sqr)
        sinr_kk = np.squeeze(sinr_num_kk) / np.squeeze(sinr_de_kk)
        rate_kk = np.log2(1 + sinr_kk)
        weighted_rate_kk = alpha[:, kk] * np.log2(1 + sinr_kk)
        rate_set.append(rate_kk)
        weighted_rate_set.append(weighted_rate_kk)
    sum_rate = np.squeeze(np.mean(np.sum(rate_set, axis=0)))
    weighted_sum_rate = np.squeeze(np.mean(np.sum(weighted_rate_set, axis=0)))
    return weighted_sum_rate, (sum_rate, direct_power_set,interference_set)


def compute_weighted_sum_rate(channel_cascaded, v, p, alpha, sigma2_dB):
    '''
    :param channel_cascaded: [num_samples,num_user_tx,num_elements_irs, num_user_rx]
    :param v: [num_samples, num_elements_irs, 1]
    :param p: [num_samples, num_user]
    :param alpha: [num_samples, num_user] weights of rate
    :param sigma2_dB: scalar in dB
    :return: average sum rate of the dataset
    '''
    sigma_sqr = 10**(sigma2_dB/10)
    # sigma_sqr = 0
    [num_samples, num_user, num_elements_irs, _] = channel_cascaded.shape
    v = np.reshape(v,[-1,num_elements_irs, 1])
    p = np.reshape(p,[-1,num_user])
    channel_cascaded = np.reshape(channel_cascaded,[-1,num_user,num_elements_irs,num_user])
    rate_set = []
    weighted_rate_set = []
    direct_power_set = []
    interference_set = []
    for kk in range(num_user):  # rx
        ch_kk = np.reshape(channel_cascaded[:, kk, :, kk], [-1, 1, num_elements_irs])
        sinr_num_kk = p[:, kk] * np.squeeze((np.abs(ch_kk @ v) ** 2))
        sinr_de_kk = sigma_sqr  # noise power
        direct_power_set.append(sinr_num_kk)
        for jj in range(num_user):
            if jj != kk:
                ch_kj = np.reshape(channel_cascaded[:, jj, :, kk], [-1, 1, num_elements_irs])
                sinr_de_kk = sinr_de_kk + p[:, jj] * np.squeeze((np.abs(ch_kj @ v) ** 2))
        interference_set.append(sinr_de_kk - sigma_sqr)
        sinr_kk = np.squeeze(sinr_num_kk) / np.squeeze(sinr_de_kk)
        rate_kk = np.log2(1 + sinr_kk)
        weighted_rate_kk = alpha[:,kk] * np.log2(1 + sinr_kk)
        rate_set.append(rate_kk)
        weighted_rate_set.append(weighted_rate_kk)
    sum_rate = np.squeeze(np.mean(np.sum(rate_set, axis=0)))
    weighted_sum_rate = np.squeeze(np.mean(np.sum(weighted_rate_set, axis=0)))

    return weighted_sum_rate, (sum_rate, direct_power_set, interference_set)


def get_cascaded_channel(channel_tx,channel_rx):
    [num_samples,num_elements_irs,num_user] = channel_tx.shape
    channel_a = np.zeros([num_samples,num_user, num_elements_irs,num_user],dtype=complex)
    for kk in range(num_user):
        for jj in range(num_user):
            channel_a[:,jj,:,kk] = channel_tx[:,:,jj]*channel_rx[:,:,kk]
    return channel_a