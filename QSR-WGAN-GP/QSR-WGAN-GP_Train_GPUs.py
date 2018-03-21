#######################################################################################################################
# Project QSRCNN: Quantized Speech Reconstruction using Convolutional Neural Network     V1.0
# =====================================================================================================================
# QSR-WGAN-GP_Train_GPUs.py: Trying to use WGAN-GP to do Quantized Speech Reconstruction
#
#
# =====================================================================================================================
# Abteilung Signalverarbeitung, IfN, Technische Universitaet Braunschweig, Deutschland
# Author:    Huijun Liu M.Sc.
# Date:      16.07.2017
#######################################################################################################################
import os
import time
import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import scipy.io.wavfile as swave
import keras.optimizers as optimizers

from numpy import random
from keras import initializers
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Add
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard

# from weightnorm import AdamWithWeightnorm
from tensorflow.python.framework import ops
from keras.backend.tensorflow_backend import set_session


#####################################################################################
# 0. Setteings For GPUs
#####################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # x stand for GPU index: 3-x!!
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # Only 30% Memory of GPUs can be used
set_session(tf.Session(config=config))

#####################################################################################
# 1. Define new Metric, Activation function and Loss function
#####################################################################################


def snr(y_true, y_pred):
    """
        SNR is Signal to Noise Ratio

    """
    return 10.0 * K.log((K.sum(K.square(y_true))) / (K.sum(K.square(y_pred - y_true)))) / K.log(10.0)


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


#####################################################################################
# 2. Define Generative model and Adversarial model
#####################################################################################
def create_generator(inputs_gen):
    n1 = 32
    n2 = 64
    n3 = 32

    c1 = Conv1D(n1, 3, padding='same', name='conv_1')(inputs_gen)
    c1 = Activation(selu, name='act_1')(c1)
    c1 = Conv1D(n1, 3, padding='same', name='conv_2')(c1)
    c1 = Activation(selu, name='act_2')(c1)
    x = MaxPooling1D(2, name='mpool_1')(c1)

    c2 = Conv1D(n2, 3, padding='same', name='conv_3')(x)
    c2 = Activation(selu, name='act_3')(c2)
    c2 = Conv1D(n2, 3, padding='same', name='conv_4')(c2)
    c2 = Activation(selu, name='act_4')(c2)
    x = MaxPooling1D(2, name='mpool_2')(c2)

    c3 = Conv1D(n3, 3, padding='same', name='conv_5')(x)
    c3 = Activation(selu, name='act_5')(c3)
    x = UpSampling1D(2, name='usample_1')(c3)

    c2_2 = Conv1D(n2, 3, padding='same', name='conv_6')(x)
    c2_2 = Activation(selu, name='act_6')(c2_2)
    c2_2 = Conv1D(n2, 3, padding='same', name='conv_7')(c2_2)
    c2_2 = Activation(selu, name='act_7')(c2_2)

    m1 = Add(name='add_1')([c2, c2_2])
    m1 = UpSampling1D(2, name='usample_2')(m1)

    c1_2 = Conv1D(n1, 3, padding='same', name='conv_8')(m1)
    c1_2 = Activation(selu, name='act_8')(c1_2)
    c1_2 = Conv1D(n1, 3, padding='same', name='conv_9')(c1_2)
    c1_2 = Activation(selu, name='act_9')(c1_2)

    m2 = Add(name='add_2')([c1, c1_2])

    decoded = Conv1D(1, 5, padding='same', activation='linear', name='conv_10')(m2)

    return decoded


def create_discriminator(inputs_disc):
    x = Conv1D(32, 3, padding='same', name='dis_conv_1')(inputs_disc)
    x = Activation(selu, name='dis_act_1')(x)

    x = Conv1D(64, 3, padding='same', name='dis_conv_2')(x)
    x = BatchNormalization(name='dis_bnorm_1')(x)
    x1 = Activation(selu, name='dis_act_2')(x)

    m1 = Add(name='dis_add_1')([inputs_disc, x1])

    x = Conv1D(32, 3, padding='same', name='dis_conv_3')(m1)
    x = Activation(selu, name='dis_act_3')(x)

    x = Conv1D(64, 3, padding='same', name='dis_conv_4')(x)
    x = BatchNormalization(name='dis_bnorm_2')(x)
    x2 = Activation(selu, name='dis_act_4')(x)
    m2 = Add(name='dis_add_2')([m1, x2])

    discri = Conv1D(1, 5, padding='same', name='dis_conv_5')(m2)

    return discri

#####################################################################################
# 3. Define Training process of QSR_WGAN_GP
#####################################################################################
SEQ_LEN = 80
BATCH_SIZE = 128


def load_data():
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
    mat_input_vali = 'Train_G711_PreProc_defautLang/Vali_inputSet_ALaw_defautLang_OLdata_ValiTrain_smallVali.mat'
    mat_input_vali = os.path.normcase(mat_input_vali)
    print('> Validation Input: ' + mat_input_vali)

    x_train_noisy_vali = sio.loadmat(mat_input_vali)
    x_train_noisy_vali = x_train_noisy_vali['inputSetNorm']
    x_train_noisy_vali = np.array(x_train_noisy_vali)

    # Load Target Data
    mat_target = 'Train_G711_PreProc_defautLang/Train_targetSet_ALaw_defautLang_OLdata_ValiTrain_Reverse_Extended_Reverse.mat'
    mat_target = os.path.normcase(mat_target)
    print('> Training Target: ' + mat_target)

    x_train = sio.loadmat(mat_target)
    x_train = x_train['targetSet']
    x_train = np.array(x_train)
    # x_train = target_min_max_scaler.fit_transform(x_train)

    # Load Target Data for Validation
    mat_target_vali = 'Train_G711_PreProc_defautLang/Vali_targetSet_ALaw_defautLang_OLdata_ValiTrain_smallVali.mat'
    mat_target_vali = os.path.normcase(mat_target_vali)
    print('> Validation Target: ' + mat_target_vali)

    x_train_vali = sio.loadmat(mat_target_vali)
    x_train_vali = x_train_vali['targetSet']
    x_train_vali = np.array(x_train_vali)

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

    x_train_noisy_vali = np.reshape(x_train_noisy_vali, (x_train_noisy_vali.shape[0], x_train_noisy_vali.shape[1], 1))
    x_train_vali = np.reshape(x_train_vali, (x_train_vali.shape[0], x_train_vali.shape[1], 1))

    print('> Data Loaded. Model Compiling... ')
    return x_train_noisy, x_train, x_train_noisy_vali, x_train_vali

# 2. dataset generator
def batch_generator(x_train_noisy, x_train, batch_size=128):
    while True:
        for i in range(0, x_train_noisy.shape[0] - batch_size + 1, batch_size):
            yield x_train_noisy[i:i+batch_size], x_train[i:i+batch_size]

default_opt_params = {'lr': 5e-5, 'clip': 1e-2, 'n_lambda': 10, 'n_critic': 10}


class QSRWGAN(object):
    def __init__(self, opt_params=default_opt_params, frame_len=80):
        self.n_critic = opt_params['n_critic']
        self.n_lambda = opt_params['n_lambda']
        self.clip = opt_params['clip']
        self.frame_len = frame_len

        # ------------------------------------------------------------------
        # 1. create session
        # ------------------------------------------------------------------
        self.sess = tf.Session()
        K.set_session(self.sess)  # pass the session to keras

        # ------------------------------------------------------------------
        # 2. create generator and discriminator
        # ------------------------------------------------------------------
        with tf.name_scope('generator'):
            gen_inputs = Input(shape=(self.frame_len, 1))
            gen_outputs = create_generator(gen_inputs)

        with tf.name_scope('discriminator'):
            dis_inputs = Input(shape=(self.frame_len, 1))
            dis_outputs = create_discriminator(dis_inputs)

        # ------------------------------------------------------------------
        # 3. instantiate networks of generator and discriminator
        # ------------------------------------------------------------------
        Generator = Model(inputs=gen_inputs, outputs=gen_outputs)
        Generator.summary()
        self.gen_model = Generator
        Discriminator = Model(inputs=dis_inputs, outputs=dis_outputs)
        Discriminator.summary()

        # ------------------------------------------------------------------
        # 4. save the inputs of generator and discriminator
        # ------------------------------------------------------------------
        quan_inputs = tf.placeholder(tf.float32, shape=(None, self.frame_len, 1), name='quan_inputs')
        real_inputs = tf.placeholder(tf.float32, shape=(None, self.frame_len, 1), name='real_inputs')
        self.inputs = quan_inputs, real_inputs

        # ------------------------------------------------------------------
        # 5. get the weights of generator and discriminator
        # ------------------------------------------------------------------
        self.gen_weights = [weights for weights in tf.global_variables() if 'generator' in weights.name]
        self.dis_weights = [weights for weights in tf.global_variables() if 'discriminator' in weights.name]
        # self.gen_weights = Generator.get_weights()
        # self.dis_weights = Discriminator.get_weights()

        # ------------------------------------------------------------------
        # 6. create predictions of generator and discriminator
        # ------------------------------------------------------------------
        fake_inputs = Generator(quan_inputs)
        disc_real = Discriminator(real_inputs)
        disc_fake = Discriminator(fake_inputs)
        self.predictions = fake_inputs

        # ------------------------------------------------------------------
        # 7. create losses and compute probabilities of discriminator
        # ------------------------------------------------------------------
        # 7.1. WGAN lipschitz-penalty
        alpha = tf.random_uniform(shape=[BATCH_SIZE, 1, 1], minval=-0.4, maxval=0.4)
        differences = fake_inputs - real_inputs
        interpolates = real_inputs + (alpha * differences)

        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        # gradient_penalty = self.n_lambda * tf.reduce_mean((slopes - 1.) ** 2)
        gp = K.mean(K.square(K.sqrt(K.sum(K.square(gradients), axis=1)) - 1))
        gradient_penalty = self.n_lambda * gp

        disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        disc_loss += gradient_penalty

        self.dis_loss = disc_loss
        self.gen_loss = -tf.reduce_mean(disc_fake)

        self.disc_real = tf.reduce_mean(disc_real)
        self.disc_fake = tf.reduce_mean(disc_fake)
        self.prob_real = tf.reduce_mean(tf.sigmoid(disc_real))
        self.prob_fake = tf.reduce_mean(tf.sigmoid(disc_fake))

        # ------------------------------------------------------------------
        # 8. create optimizer for generator and discriminator
        # ------------------------------------------------------------------
        learning_rate = opt_params['lr']

        gen_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.gen_loss, var_list=self.gen_weights)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.dis_loss, var_list=self.dis_weights)

        self.gen_opt_train = gen_train_op
        self.dis_opt_train = disc_train_op

    def load_weights(self):
        self.gen_model.load_weights('ddsrcnn_weights_defaultLang_OL40_stopstr_bs128_lr5e-05.h5')

    def save_weights(self, file_path):
        file_path = os.path.normcase(file_path)
        self.gen_model.save_weights(file_path)

    def load_batch(self, x_train_noise, x_train, train=True):
        gen_inputs, dis_inputs = self.inputs
        return {gen_inputs: x_train_noise, dis_inputs: x_train, K.learning_phase(): train}

    def gen(self, x_train_noise):
        gen_inputs, dis_inputs = self.inputs
        feed_dict = {gen_inputs: x_train_noise, K.learning_phase(): False}
        return self.sess.run(self.predictions, feed_dict=feed_dict)

    def gen_train(self, feed_dict):
        _, gen_loss = self.sess.run([self.gen_opt_train, self.gen_loss], feed_dict=feed_dict)
        return gen_loss

    def dis_train(self, feed_dict):
        # take a step of adam
        _, dis_loss = self.sess.run([self.dis_opt_train, self.dis_loss], feed_dict=feed_dict)
        # return discriminator loss
        return dis_loss

    def fit(self, x_train_noise, x_train, x_train_noise_vali, x_train_vali, epochs=10, logdir='/qsrwgan_run'):
        # ------------------------------------------------------------------
        # 1. initialize log directory
        # ------------------------------------------------------------------
        if tf.gfile.Exists(logdir):
            tf.gfile.DeleteRecursively(logdir)

        tf.gfile.MakeDirs(logdir)

        # ------------------------------------------------------------------
        # 2. initialize model
        # ------------------------------------------------------------------
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.load_weights()

        # ------------------------------------------------------------------
        # 3. train the model
        # ------------------------------------------------------------------
        step, g_step, epoch = 0, 0, 0
        curr_epoch = 0

        # create data for the gan training
        # generator = batch_generator(x_train_noise, x_train)
        mat_input = 'Train_G711_PreProc_defautLang/inputTestSet_g711concat_nonOL_Frame_80.mat'
        mat_input = os.path.normcase(mat_input)
        x_train_noisy = sio.loadmat(mat_input)
        x_train_noisy = x_train_noisy['inputTestNorm']
        x_train_noisy = np.array(x_train_noisy)
        x_train_noisy = np.reshape(x_train_noisy, (x_train_noisy.shape[0], x_train_noisy.shape[1], 1))

        while curr_epoch < epochs:
            # create data for the gan training
            generator = batch_generator(x_train_noise, x_train, BATCH_SIZE)
            # generator_vali = batch_generator(x_train_noise_vali, x_train_vali, 1024)

            curr_iter = 0
            while curr_iter < x_train_noise.shape[0]//BATCH_SIZE:
                start_time = time.time()
                # n_critic = 100 if g_step < 25 or (g_step+1) % 500 == 0 else self.n_critic

                for i in range(self.n_critic):
                    curr_iter += 1
                    dis_losses = []

                    # load the batch
                    quant_batch, real_batch = generator.__next__()
                    # quant_batch = np.random.randn(BATCH_SIZE, 80, 1)
                    feed_dict = self.load_batch(quant_batch, real_batch)

                    # train the discriminator
                    dis_loss = self.dis_train(feed_dict)
                    dis_losses.append(dis_loss)

                dis_loss = np.array(dis_losses).mean()

                # train the generator
                curr_iter += 1
                quant_batch, real_batch = generator.__next__()
                # quant_batch = np.random.randn(BATCH_SIZE, 80, 1)
                feed_dict = self.load_batch(quant_batch, real_batch)
                gen_loss = self.gen_train(feed_dict)

                g_step += 1

                if g_step < 1000 or g_step % 1000 == 0:
                    tot_time = time.time() - start_time
                    print('Epoch: %3d, Gen Steps: %4d (%3.lf s), Discriminator loss: %.6f, Generator loss: %.6f' % (curr_epoch, g_step, tot_time, dis_loss, gen_loss))

                if g_step % 50 == 0:


                    prediction = self.gen(np.random.randn(BATCH_SIZE, 80, 1))
                    # feed_dict = self.load_batch(x_train_noisy, real_batch_vali)
                    # quanspeech, realspeech = self.sess.run(self.inputs, feed_dict)
                    fname = 'recon-speech-%d_%d.wav' % (curr_iter, g_step)
                    swave.write(fname, 8000, np.reshape(prediction, (prediction.size,)))
                    # fname = 'real-speech-%d.wav' % g_step
                    # swave.write(fname, 8000, np.reshape(realspeech, (realspeech.size,)))

                    # fig = plt.figure(facecolor='white')
                    # ax = fig.add_subplot(111)
                    # ax.plot(np.reshape(realspeech, (realspeech.size,)), label='RealSpeech')
                    # plt.plot(np.reshape(quanspeech, (quanspeech.size,)), label='QuanSpeech')
                    # plt.plot(np.reshape(prediction, (prediction.size,)), label='Prediction')

                    # plt.legend()
                    # plt.show()

            curr_epoch += 1

        self.save_weights("qsrwgan_weights.h5")


#####################################################################################
# 4. QSR_WGAN_GP Training
#####################################################################################		
model = QSRWGAN(opt_params=default_opt_params)
# train model
x_train_noisy, x_train, _, _ = load_data()
model.fit(x_train_noisy, x_train, _, _, epochs=10000)
