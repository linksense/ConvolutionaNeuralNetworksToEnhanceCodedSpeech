#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as swave
from numpy import random

import tensorflow as tf
import tflib as lib
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from keras.backend.tensorflow_backend import set_session
#os.environ["CUDA_VISIBLE_DEVICES"] = "3" # x stand for GPU index: 3-x!!
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6  # Only 30% Memory of GPUs can be used
#set_session(tf.Session(config=config))


#####################################################################################
# 0. Hyperparameter Init
#####################################################################################
# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
BATCH_SIZE = 32  # Batch size
ITERS = 200000  # How many iterations to train for
SEQ_LEN = 80  # Sequence length in characters
DIM = 128  # Model dimensionality. This is fairly slow and overfits, even on
# Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 5  # How many critic iterations per generator iteration. We
# use 10 for the results in the paper, but 5 should work fine
# as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 100000  # Max number of data examples to load. If data loading
# is too slow or takes too much RAM, you can decrease
# this (at the expense of having less training data).


#####################################################################################
# 1. load data
#####################################################################################
print('> Loading data... ')
# Load Input Data
mat_input = 'Train_G711_PreProc_defautLang/Train_inputSet_ALaw_defautLang_OLdata_ValiTrain_Reverse_Extended_Reverse.mat'
mat_input = os.path.normcase(mat_input)
print('> Training Input: ' + mat_input)

x_train_noisy = sio.loadmat(mat_input)
x_train_noisy = x_train_noisy['inputSetNorm']
x_train_noisy = np.array(x_train_noisy)
# x_train_noisy = input_min_max_scaler.fit_transform(x_train_noisy)

# Load Input Data for Validation
# mat_input_vali = 'Train_G711_PreProc_defautLang/Vali_inputSet_ALaw_defautLang_OLdata_ValiTrain_smallVali_Reverse_Extended.mat'
# mat_input_vali = os.path.normcase(mat_input_vali)
# print('> Validation Input: ' + mat_input_vali)

# x_train_noisy_vali = sio.loadmat(mat_input_vali)
# x_train_noisy_vali = x_train_noisy_vali['inputSetNorm']
# x_train_noisy_vali = np.array(x_train_noisy_vali)

# Load Target Data
mat_target = 'Train_G711_PreProc_defautLang/Train_targetSet_ALaw_defautLang_OLdata_ValiTrain_Reverse_Extended_Reverse.mat'
mat_target = os.path.normcase(mat_target)
print('> Training Target: ' + mat_target)

x_train = sio.loadmat(mat_target)
x_train = x_train['targetSet']
x_train = np.array(x_train)
# x_train = target_min_max_scaler.fit_transform(x_train)

# Load Target Data for Validation
# mat_target_vali = 'Train_G711_PreProc_defautLang/Vali_targetSet_ALaw_defautLang_OLdata_ValiTrain_smallVali_Reverse_Extended.mat'
# mat_target_vali = os.path.normcase(mat_target_vali)
# print('> Validation Target: ' + mat_target_vali)

# x_train_vali = sio.loadmat(mat_target_vali)
# x_train_vali = x_train_vali['targetSet']
# x_train_vali = np.array(x_train_vali)

# Randomization of Training Pairs (Currently NO Shuffle)
random.seed(1331)

train = np.column_stack((x_train_noisy, x_train))
np.random.shuffle(train)
x_train_noisy = train[:, :SEQ_LEN]
x_train = train[:, SEQ_LEN:]

# Reshape of Traing Pairs and validation Pairs
x_train_noisy = np.reshape(x_train_noisy, (x_train_noisy.shape[0], x_train_noisy.shape[1], 1))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# validation = np.column_stack((x_train_noisy_vali, x_train_vali))
# np.random.shuffle(validation )
# x_train_noisy_vali = validation [:, :SEQ_LEN]
# x_train_vali = validation [:, SEQ_LEN:]

# x_train_noisy_vali = np.reshape(x_train_noisy_vali, (x_train_noisy_vali.shape[0], x_train_noisy_vali.shape[1], 1))
# x_train_vali = np.reshape(x_train_vali, (x_train_vali.shape[0], x_train_vali.shape[1], 1))

print('> Data Loaded. Model Compiling... ')

def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name + '.1', DIM, DIM, 5, output, biases=False)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name + '.2', DIM, DIM, 5, output, biases=False)
    return inputs + (0.3 * output)


####################################################
# 1. Define Generator Model
####################################################
def Generator(inputs):
    output = tf.transpose(inputs, [0, 2, 1])
    output = lib.ops.conv1d.Conv1D('Generator.Input', SEQ_LEN, DIM, 1, output, biases=False)
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, SEQ_LEN, 1, output, biases=False)
    output = tf.transpose(output, [0, 2, 1])
    output = tf.tanh(output)
    return output

####################################################
# 2. Define Discriminator Model
####################################################
def Discriminator(inputs):
    output = tf.transpose(inputs, [0, 2, 1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', SEQ_LEN, DIM, 1, output, biases=False)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, BATCH_SIZE * DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', BATCH_SIZE * DIM, 1, output, biases=False)
    return output

####################################################
# 3. Define inputs of all Models
####################################################
real_inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN, 1])
quan_inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN, 1])

fake_inputs = Generator(quan_inputs)

disc_real = Discriminator(real_inputs)
disc_fake = Discriminator(fake_inputs)

####################################################
# 4. WGAN lipschitz-penalty
####################################################
alpha = tf.random_uniform(shape=[BATCH_SIZE, 1, 1], minval=0., maxval=1.)
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha * differences)

# 1. loss
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
gradient_penalty = LAMBDA * tf.reduce_mean((slopes - 1.) ** 2)

disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
disc_loss += gradient_penalty
gen_loss = -tf.reduce_mean(disc_fake)

# 3. optimizer
gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')
gen_train_op = tf.train.AdamOptimizer(learning_rate=5e-6, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=5e-6, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=disc_params)

# 4. dataset generator
def batch_generator():
    while True:
        for i in range(0, x_train_noisy.shape[0] - BATCH_SIZE + 1, BATCH_SIZE):
            yield x_train_noisy[i:i+BATCH_SIZE], x_train[i:i+BATCH_SIZE]

if not os.path.exists('out/'):
    os.makedirs('out/')

#################################################################
# 5. Training Loop
#################################################################
with tf.Session() as session:
    session.run(tf.global_variables_initializer()) # variables init

    # create data for the gan training
    generator = batch_generator()

    for iteration in range(ITERS):
        start_time = time.time()

        # Train critic
        for i in range(CRITIC_ITERS):
            quant_batch, real_batch = generator.__next__()
            _, _gen_loss = session.run([gen_train_op, gen_loss], feed_dict={quan_inputs: quant_batch})
            _, _disc_loss = session.run([disc_train_op, disc_loss], feed_dict={real_inputs:real_batch, quan_inputs:quant_batch})

        # Train generator
        quant_batch, real_batch = generator.__next__()
        _, _gen_loss = session.run([gen_train_op, gen_loss], feed_dict={quan_inputs: quant_batch})

        if iteration % 10 == 0:
            print('epoch %s, disc_loss: %s, gen_loss: %s' % (iteration, _disc_loss, _gen_loss))

            realspeech, quanspeech, prediction = session.run([real_inputs, quan_inputs, fake_inputs], feed_dict={real_inputs: real_batch, quan_inputs: quant_batch})

            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.plot(np.reshape(realspeech, (realspeech.size,)), label='RealSpeech')
            plt.plot(np.reshape(quanspeech, (quanspeech.size,)), label='QuanSpeech')
            plt.plot(np.reshape(prediction, (prediction.size,)), label='Prediction')

            plt.legend()
            plt.show()

