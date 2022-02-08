import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
# from func_codedesign_cont import func_codedesign_cont
import scipy.io as sio
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense

'System Information'
N = 64  # Number of BS's antennas
delta_inv = 128  # Number of posterior intervals inputed to DNN 
delta = 1 / delta_inv
S = np.log2(delta_inv)
tau = 2  # Pilot length; To generate fig 6, we trained different networks with different tau values.
OS_rate = 20  # Over sampling rate in  each AoA interval (N_s in the paper)
delta_inv_OS = OS_rate * delta_inv  # Total number of AoAs for posterior computation
delta_OS = 1 / delta_inv_OS
'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
num_SNR = 8  # Number of considered SNRs
low_SNR_idx = 7  # Index of Lowest SNR for training
high_SNR_idx = 8  ##Index of highest SNR for training + 1
idx_SNR_val = 7  #Index of SNR for validation (saving parameters)
snrdBvec = np.linspace(start=-10,stop=25,num=num_SNR) #Set of SNRs
Pvec = 10 ** (snrdBvec / 10)  # Set of considered TX powers
mean_true_alpha = 0.0 + 0.0j  # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)  # STD of the Gaussian noise per real dim.
L = 2 #Number of paths
#####################################################
'Learning Parameters'
initial_run = 1  # 0: Continue training; 1: Starts from the scratch
n_epochs = 10000  # Num of epochs
learning_rate = 0.00008  # Learning rate
batch_per_epoch = 200  # Number of mini batches per epoch
batch_size_order = 8  # Mini_batch_size = batch_size_order*delta_inv
val_size_order = 200  # Validation_set_size = val_size_order*delta_inv
scale_factor = 1  # Scaling the number of tests
test_size_order = 200  # Test_set_size = test_size_order*delta_inv*scale_factor
model_path = './param/params_multi_AoA_est'+'_'+str(tau)
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
    hidden_size = 512
    A1 = tf.get_variable("A1", shape=[hidden_size, 1024], dtype=tf.float32, initializer=he_init)
    A2 = tf.get_variable("A2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A3 = tf.get_variable("A3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A4 = tf.get_variable("A4", shape=[1024, 2 * N], dtype=tf.float32, initializer=he_init)
    
    C1 = tf.get_variable("C1", shape=[hidden_size, 1024], dtype=tf.float32, initializer=he_init)    
    C2 = tf.get_variable("C2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    C3 = tf.get_variable("C3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    C4 = tf.get_variable("C4", shape=[1024, L], dtype=tf.float32, initializer=he_init)
    
    b1 = tf.get_variable("b1", shape=[1024], dtype=tf.float32, initializer=he_init)
    b2 = tf.get_variable("b2", shape=[1024], dtype=tf.float32, initializer=he_init)
    b3 = tf.get_variable("b3", shape=[1024], dtype=tf.float32, initializer=he_init)
    b4 = tf.get_variable("b4", shape=[2 * N], dtype=tf.float32, initializer=he_init)
    
    d1 = tf.get_variable("d1", shape=[1024], dtype=tf.float32, initializer=he_init)
    d2 = tf.get_variable("d2", shape=[1024], dtype=tf.float32, initializer=he_init)
    d3 = tf.get_variable("d3", shape=[1024], dtype=tf.float32, initializer=he_init)
    d4 = tf.get_variable("d4", shape=[L], dtype=tf.float32, initializer=he_init)

    w_dict = []
    posterior_dict = []
    idx_est_dict = []
    layer_Ui = Dense(units=hidden_size, activation='linear')
    layer_Wi = Dense(units=hidden_size, activation='linear')
    layer_Uf = Dense(units=hidden_size, activation='linear')
    layer_Wf = Dense(units=hidden_size, activation='linear')
    layer_Uo = Dense(units=hidden_size, activation='linear')
    layer_Wo = Dense(units=hidden_size, activation='linear')
    layer_Uc = Dense(units=hidden_size, activation='linear')
    layer_Wc = Dense(units=hidden_size, activation='linear')
    layer_V2 = Dense(units=delta_inv, activation='linear')

    def RNN(input_x, h_old, c_old):
        i_t = tf.sigmoid(layer_Ui(input_x) + layer_Wi(h_old))
        f_t = tf.sigmoid(layer_Uf(input_x) + layer_Wf(h_old))
        o_t = tf.sigmoid(layer_Uo(input_x) + layer_Wo(h_old))
        c_t = tf.tanh(layer_Uc(input_x) + layer_Wc(h_old))
        c = i_t * c_t + f_t * c_old
        h_new = o_t * tf.tanh(c)
        return h_new, c

    snr = lay['P'] * tf.ones(shape=[batch_size, 1], dtype=tf.float32)
    snr_dB = tf.log(snr) / np.log(10)
    snr_normal = (snr_dB-1)/np.sqrt(1.6666) #Normalizing for the range -10dB to 30dB

    bf_gain_list = []
    for t in range(tau):
        'DNN designs the next sensing direction'
        if t == 0:
            y_abs = tf.ones([batch_size, 1])
            h_old = tf.zeros([batch_size, hidden_size])
            c_old = tf.zeros([batch_size, hidden_size])
        alpha_real = tf.reshape(tf.concat([tf.real(alpha_input),tf.imag(alpha_input)],axis=-1),(-1,2*L))
        h_old, c_old = RNN(tf.concat([y_abs, snr_normal], axis=1), h_old, c_old)

        x1 = tf.nn.relu(h_old @ A1 + b1)
        x1 = BatchNormalization()(x1)
        x2 = tf.nn.relu(x1 @ A2 + b2)
        x2 = BatchNormalization()(x2)
        x3 = tf.nn.relu(x2 @ A3 + b3)
        x3 = BatchNormalization()(x3)
        w_her = x3 @ A4 + b4
        # w_her = x1@A4+b4

        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm)
        w_her_complex = tf.complex(w_her[:, 0:N], w_her[:, N:2 * N])
        w_dict.append(w_her_complex)
        W_her = tf.stack(w_dict, axis=1)
        'BS observes the next measurement'
        y_noiseless = tf.reduce_sum(tf.multiply(w_her_complex, a_phi), 1, keepdims=True)
        noise = tf.complex(tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=noiseSTD_per_dim), \
                           tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=noiseSTD_per_dim))
        # y_complex = tf.tile(tf.complex(tf.sqrt(lay['P']),0.0)*tf.multiply(y_noiseless,alpha_input) + noise,(1,delta_inv))
        y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless + noise
        #y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1) / tf.sqrt(lay['P'])
        y_abs = tf.abs(y_complex)/tf.sqrt(lay['P'])
        
    h_old, c_old = RNN(tf.concat([y_abs, snr_normal], axis=1), h_old, c_old)    
    z1 = tf.nn.relu(c_old @ C1 + d1)
    z1 = BatchNormalization()(z1)
    z2 = tf.nn.relu(z1 @ C2 + d2)
    z2 = BatchNormalization()(z2)
    z3 = tf.nn.relu(z2 @ C3 + d3)
    z3 = BatchNormalization()(z3)
    output_phi = z3 @ C4 + d4
#        bf_gain_list.append(bf_gain)
        ####################################################################################
####### Loss Function
loss = tf.reduce_mean(tf.reduce_sum((output_phi-phi_input)**2,axis=-1))
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
sorted_idx_val = np.argsort(np.abs(alpha_val),axis=1)           
alpha_val =  np.take_along_axis(alpha_val, sorted_idx_val, axis=1)       
phi_val = np.random.uniform(low=phi_min, high=phi_max, size=[delta_inv * val_size_order,L])
feed_dict_val = {alpha_input: alpha_val,
                 phi_input: phi_val,
                 lay['P']: Pvec[idx_SNR_val]}
###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, model_path)
        # saver.restore(sess, './param/params_multi_AoA_BF'+str(snrdBvec[idx_SNR_val])+'_'+str(20))
    best_loss, pp = sess.run([loss, posterior_dict], feed_dict=feed_dict_val)
    print(best_loss)
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            idx_temp = np.random.randint(low=low_SNR_idx, high=high_SNR_idx, size=1)
            snr_temp = snrdBvec[idx_temp[0]]
            # snr_temp = snrdBvec[idx_SNR_val]
            P_temp = 10 ** (snr_temp / 10)
            alpha_batch = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                           size=[batch_size_order * delta_inv, L]) \
                          + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                                  size=[batch_size_order * delta_inv, L])
            sorted_idx_batch = np.argsort(np.abs(alpha_batch),axis=1)           
            alpha_batch =  np.take_along_axis(alpha_batch, sorted_idx_batch, axis=1)              
                          
            phi_batch = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_order * delta_inv, L])
            feed_dict_batch = {alpha_input: alpha_batch,
                               phi_input: phi_batch,
                               lay['P']: P_temp}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1

        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch', epoch, '  loss_test:%2.5f' % (loss_val), '  best_test:%2.5f' % (best_loss))
        if epoch % 10 == 9:  # Every 10 iterations it checks if the validation performace is improved, then saves parameters  
            if loss_val < best_loss:
                save_path = saver.save(sess, model_path)
                best_loss = loss_val

    ##########  Final Test    
    performance = np.zeros([len(snrdBvec), scale_factor])
    for j in range(scale_factor):
        print(j)
        alpha_test = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                       size=[test_size_order * delta_inv, L]) \
                      + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                              size=[test_size_order * delta_inv, L])
        sorted_idx_test = np.argsort(np.abs(alpha_test),axis=1)           
        alpha_test =  np.take_along_axis(alpha_test, sorted_idx_test, axis=1)               
        phi_test = np.random.uniform(low=phi_min, high=phi_max, size=[delta_inv * val_size_order,L])
        for i in range(len(snrdBvec)):
             feed_dict_test = {alpha_input: alpha_test,
                               phi_input: phi_test,
                               lay['P']: Pvec[i]}
             mse_loss = sess.run(loss, feed_dict=feed_dict_test)
             performance[i, j] = mse_loss
    
    performance = np.mean(performance, axis=1)

######## Plot the test result 
plt.semilogy(snrdBvec, performance)
plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel('Average MSE')
#
sio.savemat('data_RNN_multiAoAs_nonCoh_tau2.mat', dict(performance=performance, \
                                          snrdBvec=snrdBvec, N=N, delta_inv=delta_inv, \
                                          mean_true_alpha=mean_true_alpha, \
                                          std_per_dim_alpha=std_per_dim_alpha, \
                                          noiseSTD_per_dim=noiseSTD_per_dim, tau=tau))



