import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
from generate_channel import main_generate_channel
import scipy.io as sio
from keras.layers import BatchNormalization
# from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense
from util_fun import get_cascaded_channel
import os

'This code generates the results for DNN-based design (learned sensing vectors, fixed)'

'System information'
tau = 7 # To generate fig 9, we trained different networks with different tau values.
num_ris_elements = 64
num_user = 1
irs_Nh = 8
Rician_factor = 10
scale_factor = 120

'Channel Information'
idx_SNR_val = 0
snrdBvec = np.array([0])
Pvec = 10**(snrdBvec/10)
noiseSTD_per_dim = np.sqrt(0.5) #STD of the Gaussian noise per real dim.
#####################################################
'Learning Parameters'
initial_run = 1 #0: Continue training; 1: Starts from the scratch
n_epochs = 10000 #Num of epochs
learning_rate = 0.0001 #Learning rate
batch_per_epoch = 100 #Number of mini batches per epoch
batch_size_train = 1024
batch_size_test = 10000
batch_size_val = 10000
######################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Define initialization method
######################################## Place Holders
channel_cascaded_input = tf.placeholder(tf.complex64, shape=(None,num_ris_elements), name="channel_cascaded_input")
batch_size = tf.shape(channel_cascaded_input)[0]
lay = {}
lay['P'] = tf.constant(1.0)

with tf.name_scope("Uplink_pilots_transmission"):
    for t in range(tau):
        real_v0 = tf.get_variable('real_v0'+str(t),shape=(num_ris_elements,1))
        imag_v0 = tf.get_variable('imag_v0'+str(t),shape=(num_ris_elements,1))
        real_v = real_v0/tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
        imag_v = imag_v0/tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
        w_her_complex = tf.complex(real_v,imag_v)
        w_her_complex = tf.reshape(tf.tile(w_her_complex,(batch_size,1)),[batch_size,num_ris_elements])

        'BS observes the next measurement'
        y_noiseless = tf.reduce_sum( tf.multiply(w_her_complex, channel_cascaded_input), 1, keepdims=True )
        noise =  tf.complex(tf.random_normal(tf.shape(y_noiseless), mean = 0.0, stddev = noiseSTD_per_dim),\
                    tf.random_normal(tf.shape(y_noiseless), mean = 0.0, stddev = noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']),0.0)*y_noiseless + noise
        y_real = tf.concat([tf.real(y_complex),tf.imag(y_complex)],axis=1)/tf.sqrt(lay['P'])
        if t==0:
            y_real_all = y_real
        else:
            y_real_all = tf.concat([y_real_all,y_real],axis=1)

v_dict= []
with tf.name_scope('Downlink_design'):
    A1 = tf.get_variable("A1", shape=[tau * 2, 1024], dtype=tf.float32, initializer=he_init)
    A2 = tf.get_variable("A2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A3 = tf.get_variable("A3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A4 = tf.get_variable("A4", shape=[1024, 2 * num_ris_elements], dtype=tf.float32, initializer=he_init)

    b1 = tf.get_variable("b1", shape=[1024], dtype=tf.float32, initializer=he_init)
    b2 = tf.get_variable("b2", shape=[1024], dtype=tf.float32, initializer=he_init)
    b3 = tf.get_variable("b3", shape=[1024], dtype=tf.float32, initializer=he_init)
    b4 = tf.get_variable("b4", shape=[2 * num_ris_elements], dtype=tf.float32, initializer=he_init)

    x1 = tf.nn.relu(y_real_all @ A1 + b1)
    x1 = BatchNormalization()(x1)
    x2 = tf.nn.relu(x1 @ A2 + b2)
    x2 = BatchNormalization()(x2)
    x3 = tf.nn.relu(x2 @ A3 + b3)
    x3 = BatchNormalization()(x3)
    v_down = x3 @ A4 + b4

    real_v0 = tf.reshape(v_down[:, 0:num_ris_elements], [batch_size, num_ris_elements])
    imag_v0 = tf.reshape(v_down[:, num_ris_elements:2 * num_ris_elements], [batch_size, num_ris_elements])
    real_v = real_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
    imag_v = imag_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
    v_complex = tf.complex(real_v, imag_v)
    v_complex = tf.reshape(v_complex, [batch_size, num_ris_elements])
    v_dict.append(v_complex)
    y_noiseless_d = tf.reduce_sum(tf.multiply(v_complex, channel_cascaded_input), 1, keepdims=True)
    received_power = tf.reduce_mean(tf.abs(y_noiseless_d)**2)
####################################################################################
####### Loss Function
loss = -received_power
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  Validation Set
channel_tx_val, channel_rx_val = main_generate_channel(num_ris_elements, num_user, irs_Nh, batch_size_val,
                                                         Rician_factor, location_user = None, scale_factor = 120,
                                                         isTesting = True)
h_input_val = get_cascaded_channel(channel_tx_val, channel_rx_val)
h_input_val = np.squeeze(h_input_val)
feed_dict_val = {channel_cascaded_input: h_input_val,
                  lay['P']: Pvec[idx_SNR_val]}

optimal_v_val = np.conj(h_input_val / np.abs(h_input_val))
y_tmp_val = np.sum(np.multiply(optimal_v_val, h_input_val), axis=1)
optimal_power_val = np.mean(np.abs(y_tmp_val) ** 2)
###########  Training
snr_temp = snrdBvec[idx_SNR_val]
no_increase = 0
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './params/non_addaptive_reflection_snr_trainable'+str(int(snr_temp))+'_'+str(tau))
    best_loss = sess.run(loss, feed_dict=feed_dict_val)
    print(-best_loss)
    print(-best_loss,10*np.log10(-best_loss),10*np.log10(optimal_power_val))
    print(tf.test.is_gpu_available()) #Prints whether or not GPU is on
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            P_temp = 10**(snr_temp/10)
            channel_tx_batch, channel_rx_batch = main_generate_channel(num_ris_elements, num_user, irs_Nh, batch_size_train,
                                                                   Rician_factor, location_user=None, scale_factor=120,
                                                                   isTesting=False)
            h_input_batch = get_cascaded_channel(channel_tx_batch, channel_rx_batch)
            h_input_batch = np.squeeze(h_input_batch)
            feed_dict_batch = {channel_cascaded_input: h_input_batch,
                             lay['P']:P_temp}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1
        
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch', epoch, '  loss_test:%2.5f' % -loss_val, '  optimal:%2.5f' % optimal_power_val,
              '  best_test:%2.5f  ' % -best_loss, -best_loss / optimal_power_val, '   dB:', 10 * np.log10(-loss_val),
              'no_increase:', no_increase)
        # if epoch%10==9: #Every 10 iterations it checks if the validation performace is improved, then saves parameters
        if loss_val < best_loss:
            save_path = saver.save(sess, './params/non_addaptive_reflection_snr_trainable'+str(int(snr_temp))+'_'+str(tau))
            best_loss = loss_val
            no_increase=0
        else:
            no_increase=no_increase+1
        if no_increase>20:
            break


