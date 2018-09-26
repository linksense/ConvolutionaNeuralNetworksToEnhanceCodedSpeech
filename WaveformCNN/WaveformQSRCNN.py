#########################################################################################
# CepstralQSRCNN.py: QSRCNN for G711/ADPCM/AMR/EVS using Cepstral features
# Author:   Huijun Liu
# Time:     17.07.2017
# Location: TU Braunschweig IfN
#########################################################################################

import os
import time
import math
import scipy.io as sio
import tensorflow as tf

from keras.models import Model
from keras import backend as K
from keras.layers import Input, Add, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler

from weightnorm import AdamWithWeightnorm
from tensorflow.python.framework import ops

# -------------------------------------------------------------------------------
# 0. define metric and activation function
# -------------------------------------------------------------------------------


def snr(y_true, y_pred):
    """
        SNR is Signal to Noise Ratio

    """
    return 10.0 * K.log((K.sum(K.square(y_true))) / (K.sum(K.square(y_pred - y_true)))) / K.log(10.0)


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

"""
def step_decay(epoch):
    initial_lrate = 0.001

    drop = 0.25
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate
"""

# -------------------------------------------------------------------------------
# 1. define Cepstral-QSRCNN Model
# -------------------------------------------------------------------------------


class WaveformQSRCNN(object):
    def __init__(self, opt_params={'lr': 5e-4, 'batch_size': 32, 'nb_epochs': 100},
                 model_params={'n1': 16, 'n2': 32, 'n3': 16, 'frame_len': 32},
                 codec_type_params={'weights_dir': "./model_weights", 'logdir': "./log"}):
        self.learning_rate = opt_params['lr']
        self.batch_size = opt_params['batch_size']
        self.nb_epochs = opt_params['nb_epochs']

        self.log_dir = codec_type_params['logdir']
        if not (os.path.exists(self.log_dir)):
            os.makedirs(self.log_dir)

        self.weights_dir = codec_type_params['weights_dir']
        if not (os.path.exists(self.weights_dir)):
            os.makedirs(self.weights_dir)

        self.frame_len = model_params['frame_len']
        self.model = self.create_model(model_params)

    # -------------------------------------------------------------------------------
    # Load the Weights of the Model
    # -------------------------------------------------------------------------------
    def load_weights(self, file_path=""):
        if file_path == "":
            file_path = self.weights_dir + '/' + 'G711_WaveformQSRCNN_Weights_Best_bs' + \
                           str(self.batch_size) + '_lr' + str(self.learning_rate) + '.h5'

        file_path = os.path.normcase(file_path)
        self.model.load_weights(file_path)

    # -------------------------------------------------------------------------------
    # Save the Weights of the Model
    # -------------------------------------------------------------------------------
    def save_weights(self):
        file_path = self.weights_dir + '/' + 'G711_WaveformQSRCNN_Weights_Final_bs' + \
                       str(self.batch_size) + '_lr' + str(self.learning_rate) + '.h5'
        file_path = os.path.normcase(file_path)
        self.model.save_weights(file_path)

    # -------------------------------------------------------------------------------
    # 1. define model
    # -------------------------------------------------------------------------------
    def create_model(self, model_params={'n1': 32, 'n2': 64, 'n3': 32, 'frame_len': 80}):
        frame_len = self.frame_len
        n1 = model_params['n1']
        n2 = model_params['n2']
        n3 = model_params['n3']

        input_sque = Input(shape=(frame_len, 1))
        c1 = Conv1D(n1, 3, padding='same')(input_sque)
        c1 = Activation(selu)(c1)
        c1 = Conv1D(n1, 3, padding='same')(c1)
        c1 = Activation(selu)(c1)
        x = MaxPooling1D(2)(c1)

        c2 = Conv1D(n2, 3, padding='same')(x)
        c2 = Activation(selu)(c2)
        c2 = Conv1D(n2, 3, padding='same')(c2)
        c2 = Activation(selu)(c2)
        x = MaxPooling1D(2)(c2)

        c3 = Conv1D(n3, 3, padding='same')(x)
        c3 = Activation(selu)(c3)
        x = UpSampling1D(2)(c3)

        c2_2 = Conv1D(n2, 3, padding='same')(x)
        c2_2 = Activation(selu)(c2_2)
        c2_2 = Conv1D(n2, 3, padding='same')(c2_2)
        c2_2 = Activation(selu)(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling1D(2)(m1)

        c1_2 = Conv1D(n1, 3, padding='same')(m1)
        c1_2 = Activation(selu)(c1_2)
        c1_2 = Conv1D(n1, 3, padding='same')(c1_2)
        c1_2 = Activation(selu)(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Conv1D(1, 5, padding='same', activation='linear')(m2)

        model = Model(input_sque, decoded)
        model.summary()

        learning_rate = self.learning_rate
        # adam = optimizers.Adam(lr=learning_rate)
        # model.compile(optimizer=adam, loss='mse', metrics=[SNRLoss])

        adam_wn = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam_wn, loss='mse', metrics=[snr])

        return model

    # -------------------------------------------------------------------------------
    # 2. Fit the model
    # -------------------------------------------------------------------------------
    def step_decay(self, epoch):
        initial_lrate = self.learning_rate

        drop = 0.25
        epochs_drop = 4.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lrate)
        lrate = K.get_value(self.model.optimizer.lr)
        print("> Ir reduced from %f to %f" % (old_lr, lrate))
        return lrate

    def fit(self, x_train_noisy, x_train, x_train_noisy_vali, x_train_vali):
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
        print("> Training model ...")

        nb_epochs = self.nb_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate

        # ---------------------------------------------------------
        # 1. define callback functions
        # ---------------------------------------------------------
        # Stop training after 10 epoches if the vali_loss not decreasing
        stop_str = EarlyStopping(monitor='val_snr', patience=16, verbose=1, mode='max')

        # Reduce learning rate when stop improving lr = lr*factor
        reduce_LR = ReduceLROnPlateau(monitor='val_snr', factor=0.5, patience=2, verbose=1, mode='max', epsilon=0.0001, cooldown=0, min_lr=0)

        best_weights = self.weights_dir + '/' + 'G711_WaveformQSRCNN_Weights_Best_bs' + \
                       str(batch_size) + '_lr' + str(learning_rate) + '.h5'
        best_weights = os.path.normcase(best_weights)
        model_save = ModelCheckpoint(best_weights, monitor='val_snr', save_best_only=True, mode='max', save_weights_only=True, period=1)

        logger_name = self.log_dir + '/' + 'G711_WaveformQSRCNN_log_bs' + \
                      str(batch_size) + '_lr' + str(learning_rate) + '.csv'
        logger_name = os.path.normcase(logger_name)
        logger = CSVLogger(logger_name, separator=',', append=False)
        tensor_board = TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        lrate = LearningRateScheduler(self.step_decay)

        start = time.time()

        # ---------------------------------------------------------
        # 2. fit the model
        # ---------------------------------------------------------
        print("> Training model " + "using Batch-size: " + str(batch_size) + ", Learning_rate: " + str(learning_rate) + "...")
        hist = self.model.fit(x_train_noisy, x_train, epochs=nb_epochs, batch_size=batch_size, shuffle=True,
                              validation_data=[x_train_noisy_vali, x_train_vali],
                              callbacks=[lrate, reduce_LR, stop_str, model_save, logger])

        print("> Saving Completed, Time : ", time.time() - start)
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
        return hist

    # -------------------------------------------------------------------------------
    # 3. Save loss snr val_loss val_snr as .mat File
    # -------------------------------------------------------------------------------
    def save_training_curves(self, hist):
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
        print("> Saving Training and Validation loss-metric curve ...")

        start = time.time()

        trian_curve_root = "./Opting_Results"
        if not(os.path.exists(trian_curve_root)):
            os.makedirs(trian_curve_root)
        # ---------------------------------------------------------
        # 1. Saving Training Loss
        # ---------------------------------------------------------
        TrainLossVec = trian_curve_root + '/' + 'G711_WaveformQSRCNN_TrainLoss_bs' + \
                       str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        TrainLossVec = os.path.normcase(TrainLossVec)

        sio.savemat(TrainLossVec, {'Train_Loss_Vec': hist.history['loss']})

        # ---------------------------------------------------------
        # 2. Saving Training Metric
        # ---------------------------------------------------------
        TrainSNRVec = trian_curve_root + '/' + 'G711_WaveformQSRCNN_TrainMetrice_bs' + \
                      str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        TrainSNRVec = os.path.normcase(TrainSNRVec)
        sio.savemat(TrainSNRVec, {'Train_SNR_Vec': hist.history['snr']})  # snr

        # ---------------------------------------------------------
        # 3. Saving Validation Loss
        # ---------------------------------------------------------
        ValiLossVec = trian_curve_root + '/' + 'G711_WaveformDDQSRCNN_ValiLoss_bs' + \
                      str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        ValiLossVec = os.path.normcase(ValiLossVec)
        sio.savemat(ValiLossVec, {'Vali_Loss_Vec': hist.history['val_loss']})

        # ---------------------------------------------------------
        # 4. Saving Validation Metric
        # ---------------------------------------------------------
        ValiSNRVec = trian_curve_root + '/' + 'G711_WaveformQSRCNN_ValiMetrice_bs' + \
                     str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        ValiSNRVec = os.path.normcase(ValiSNRVec)
        sio.savemat(ValiSNRVec, {'Vali_SNR_Vec': hist.history['val_snr']})  # val_snr

        print("> Saving Completed, Time : ", time.time() - start)
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

    # -------------------------------------------------------------------------------
    # 4. Evaluate the Trained Model
    # -------------------------------------------------------------------------------
    def evaluation_model(self, x_test_noisy, weights_path=""):
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
        print("> Evaluation of the Trained Model ...")
        # ---------------------------------------------------------
        # 1. Load Model Weights
        # ---------------------------------------------------------
        print('> 1. Loading the Weights of the Model ...')
        self.load_weights(weights_path)

        # ---------------------------------------------------------
        # 2. Evaluate the Model
        # ---------------------------------------------------------
        start = time.time()
        print('> 2. Evaluating the Model, Please wait for a Moment ...')
        predicted = self.model.predict(x_test_noisy)
        print('> 2. Evaluating Completed, Time : ' + str(time.time() - start))

        # ---------------------------------------------------------
        # 3. Saving the Evaluation Result
        # ---------------------------------------------------------
        print('> 3. Saving the Evaluation Result ...')
        start = time.time()
        pre_file_root = "./Test_Outputs"
        if not (os.path.exists(pre_file_root)):
            os.makedirs(pre_file_root)

        preOutput = pre_file_root + "/" + "G711_CNN_testplan_vec.mat"
        preOutput = os.path.normcase(preOutput)

        sio.savemat(preOutput, {'predictions': predicted})
        print('> 3. Evaluation Result Saving Completed, Time : ' + str(time.time() - start))
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
