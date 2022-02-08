import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.io as sio
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense

'This code generate the results for DNN-based design (random sensing vectors, fixed)'

'System Information'
N = 64  # Number of BS's antennas
tau = 11  # Pilot length; To generate fig 7, we trained different networks with different tau values.

'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
num_SNR = 7  # Number of considered SNRs
idx_SNR_val = 4  #Index of SNR for validation (saving parameters)
snrdBvec = np.linspace(start=-20,stop=10,num=num_SNR) #Set of SNRs
Pvec = 10 ** (snrdBvec / 10)  # Set of considered TX powers
mean_true_alpha = 0.0 + 0.0j  # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)  # STD of the Gaussian noise per real dim.
L = 3
#####################################################
'Learning Parameters'
initial_run = 1  # 0: Continue training; 1: Starts from the scratch
n_epochs = 10000  # Num of epochs
learning_rate = 0.0001  # Learning rate

batch_per_epoch = 200  # Number of mini batches per epoch
delta_inv = 128
batch_size_order = 8  # Mini_batch_size = batch_size_order*delta_inv
val_size_order = 200  # Validation_set_size = val_size_order*delta_inv
scale_factor = 1  # Scaling the number of tests
test_size_order = 200  # Test_set_size = test_size_order*delta_inv*scale_factor

######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################################
# Constructing the array responses for AoA samples
alpha_input = tf.placeholder(tf.complex64, shape=(None,L), name="alpha_input")
phi_input = tf.placeholder(tf.float32, shape=(None,L), name="phi_input")
batch_size = tf.shape(alpha_input)[0]
##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    for ii in range(L):
        phi_1 = tf.reshape(phi_input[:,ii],[-1,1])
        from0toN = tf.cast(tf.range(0, N, 1),tf.float32)
        phi_expanded = tf.tile(phi_1,(1,N))
        a_phi_i = tf.reshape(alpha_input[:,ii],(-1,1))*(tf.exp(1j*np.pi*tf.cast(tf.multiply(tf.sin(phi_expanded),from0toN),tf.complex64)))
        if ii==0:
            a_phi = a_phi_i
        else:
            a_phi = a_phi+a_phi_i
    a_phi = a_phi/np.sqrt(L)

with tf.name_scope("channel_sensing"):
    A1 = tf.get_variable("A1", shape=[2*tau, 1024], dtype=tf.float32, initializer=he_init)
    A2 = tf.get_variable("A2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A3 = tf.get_variable("A3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A4 = tf.get_variable("A4", shape=[1024, 2 * N], dtype=tf.float32, initializer=he_init)

    b1 = tf.get_variable("b1", shape=[1024], dtype=tf.float32, initializer=he_init)
    b2 = tf.get_variable("b2", shape=[1024], dtype=tf.float32, initializer=he_init)
    b3 = tf.get_variable("b3", shape=[1024], dtype=tf.float32, initializer=he_init)
    b4 = tf.get_variable("b4", shape=[2 * N], dtype=tf.float32, initializer=he_init)

    w_dict = []
    posterior_dict = []
    idx_est_dict = []

    snr = lay['P'] * tf.ones(shape=[batch_size, 1], dtype=tf.float32)
    snr_dB = tf.log(snr) / np.log(10)
    snr_normal = (snr_dB + 0.5) / np.sqrt(0.75)  # Normalizing for the range -10dB to 30dB

    W_uplink_np = np.random.normal(size=(1,tau, N), loc=0,scale=np.sqrt(0.5))+1j*np.random.normal(size=(1,tau, N), loc=0,scale=np.sqrt(0.5))
    W_uplink_np = W_uplink_np/np.linalg.norm(W_uplink_np,axis=-1,keepdims=True)
    W_uplink = tf.get_variable("W_uplink", dtype=tf.complex64, initializer=W_uplink_np.astype(np.complex64), trainable=False)
    y_noiseless = W_uplink @ tf.reshape(a_phi,(-1,N,1))
    noise = tf.complex(tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=noiseSTD_per_dim), \
                         tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=noiseSTD_per_dim))
    y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless + noise
    y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1) / tf.sqrt(lay['P'])
    y_real = tf.reshape(y_real,(-1,tau*2))

    x1 = tf.nn.relu(y_real @ A1 + b1)
    x1 = BatchNormalization()(x1)
    x2 = tf.nn.relu(x1 @ A2 + b2)
    x2 = BatchNormalization()(x2)
    x3 = tf.nn.relu(x2 @ A3 + b3)
    x3 = BatchNormalization()(x3)
    w_her_data = x3 @ A4 + b4
    w_norm_data = tf.reshape(tf.norm(w_her_data, axis=1), (-1, 1))
    w_her_data = tf.divide(w_her_data, w_norm_data)
    w_her_complex_data = tf.complex(w_her_data[:, 0:N], w_her_data[:, N:2 * N])
    y_noiseless_data = tf.reduce_sum(tf.multiply(w_her_complex_data, a_phi), 1, keepdims=True)
    bf_gain = tf.reduce_mean(tf.abs(y_noiseless_data)**2,axis=0)
        ####################################################################################
####### Loss Function
loss = -bf_gain
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  Validation Set
alpha_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                             size=[val_size_order * delta_inv, L]) \
            + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                    size=[val_size_order * delta_inv, L])
phi_val = np.random.uniform(low=phi_min, high=phi_max, size=[delta_inv * val_size_order,L])
feed_dict_val = {alpha_input: alpha_val,
                 phi_input: phi_val,
                 lay['P']: Pvec[idx_SNR_val]}

from0toN1 = np.float32(list(range(0, N)))
a_phi1 = 0
for ii in range(L):
    a_phi1 = a_phi1+ np.reshape(alpha_val[:, ii], [-1, 1, 1])*np.exp(1j * np.pi * from0toN1.reshape(-1,N,1) * np.sin(phi_val[:,ii]).reshape(-1,1,1))
a_phi1 = a_phi1/np.sqrt(L)
a_phi1 = np.reshape(a_phi1,(-1,N,1))
v_her_opt = a_phi1
v_her_opt = v_her_opt / np.linalg.norm(v_her_opt, axis=1, keepdims=True)
tmp_opt = np.transpose(np.conj(v_her_opt), axes=[0, 2, 1]) @ a_phi1
bf_gain_opt = np.mean(np.abs(tmp_opt) ** 2)

###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './param/params_multi_AoA_BF_DNN'+str(snrdBvec[idx_SNR_val])+'_'+str(tau))
    best_loss, pp = sess.run([loss, posterior_dict], feed_dict=feed_dict_val)
    print(10*np.log10(-best_loss),10*np.log10(bf_gain_opt))
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            snr_temp = snrdBvec[idx_SNR_val]
            P_temp = 10 ** (snr_temp / 10)
            alpha_batch = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                           size=[batch_size_order * delta_inv, L]) \
                          + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                                  size=[batch_size_order * delta_inv, L])
            phi_batch = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_order * delta_inv, L])
            feed_dict_batch = {alpha_input: alpha_batch,
                               phi_input: phi_batch,
                               lay['P']: P_temp}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1

        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch', epoch, '  loss_test:%2.5f' % -loss_val, '  best_test:%2.5f' % -best_loss)
        if epoch % 10 == 9:  # Every 10 iterations it checks if the validation performace is improved, then saves parameters  
            if loss_val < best_loss:
                save_path = saver.save(sess, './param/params_multi_AoA_BF_DNN'+str(snrdBvec[idx_SNR_val])+'_'+str(tau))
                best_loss = loss_val




