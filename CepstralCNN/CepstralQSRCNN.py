#######################################################################################################################
# Project QSRCNN: Quantized Speech Reconstruction using Convolutional Neural Network     V1.0
# =====================================================================================================================
# CepstralQSRCNN.py: QSRCNN for G711/ADPCM/AMR/EVS using Cepstral features
#
#
# =====================================================================================================================
# Technische Universität Braunschweig, IfN
# Author:    Huijun Liu M.Sc.
# Date:      17.06.2017
#######################################################################################################################

import os
import time
import math
import scipy.io as sio
import tensorflow as tf

from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Add, Multiply, Average, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
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
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

"""
def step_decay(epoch):
    initial_lrate = 0.001

    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate
"""

"""
''' TensorFlow Backend Function '''
def phase_shift(I, r):
    ''' Function copied as is from https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py'''

    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))

def depth_to_scale(input, scale, channels):
    if channels > 1:
        Xc = tf.split(3, 3, input)
        X = tf.concat(3, [phase_shift(x, scale) for x in Xc])
    else:
        X = phase_shift(input, scale)
    return X


'''
Implementation is incomplete. Use lambda layer for now.
'''
class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = depth_to_scale(x, self.r, self.channels)
        return y

    def get_output_shape_for(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return (b, self.channels, r * self.r, c * self.r)
        else:
            b, r, c, k = input_shape
            return (b, r * self.r, c * self.r, self.channels)
"""

# -------------------------------------------------------------------------------
# 1. define Cepstral-QSRCNN Model
# -------------------------------------------------------------------------------
class CepstralQSRCNN(object):
    def __init__(self, opt_params={'lr': 5e-4, 'batch_size': 32, 'nb_epochs': 100},
                 model_params={'n1': 16, 'n2': 32, 'n3': 16, 'frame_len': 32},
                 codec_type_params={'codec': 'ADPCM', 'type': '3', 'weights_dir': "./model_weights", 'logdir': "./log"}):
        self.learning_rate = opt_params['lr']                   # Learning rate
        self.batch_size = opt_params['batch_size']              # Batch size
        self.nb_epochs = opt_params['nb_epochs']                # Number of epochs

        self.codec = codec_type_params['codec']                 # Codec type
        self.type = codec_type_params['type']                   # Methods type

        self.log_dir = codec_type_params['logdir']              # Log file direction
        if not (os.path.exists(self.log_dir)):
            os.makedirs(self.log_dir)

        self.weights_dir = codec_type_params['weights_dir']     # Weights file direction
        if not (os.path.exists(self.weights_dir)):
            os.makedirs(self.weights_dir)

        self.frame_len = model_params['frame_len']              # Frame length
        self.model_params = model_params
        self.model = self.create_model("qsrcnn")

    # -------------------------------------------------------------------------------
    # Load the Weights of the Model
    # -------------------------------------------------------------------------------
    def load_weights(self, file_path=""):
        if file_path == "":
            file_path = self.weights_dir + '/' + self.codec + '_Type' + self.type + '_CepstralQSRCNN_Weights_Best_bs' + \
                           str(self.batch_size) + '_lr' + str(self.learning_rate) + '.h5'

        file_path = os.path.normcase(file_path)
        self.model.load_weights(file_path)

    # -------------------------------------------------------------------------------
    # Save the Weights of the Model
    # -------------------------------------------------------------------------------
    def save_weights(self):
        file_path = self.weights_dir + '/' + self.codec + '_Type' + self.type + '_CepstralQSRCNN_Weights_Final_bs' + \
                       str(self.batch_size) + '_lr' + str(self.learning_rate) + '.h5'
        file_path = os.path.normcase(file_path)
        self.model.save_weights(file_path)

	"""
    def _upscale_block(self, ip, id):
        init = ip

        x = Conv1D(256, 3, padding='same', name='espcnn_upconv1_%d' % id)(init)
        x = Activation(selu)(x)
        x = SubPixelUpscaling(r=2, channels=64, name='espcnn_upconv1__upscale1_%d' % id)(x)
        x = Conv1D(256, 3, padding='same', name='espcnn_upconv1_filter1_%d' % id)(x)
        x = Activation(selu)(x)

        return x
	"""
	
    # -------------------------------------------------------------------------------
    # 1. define model
    # -------------------------------------------------------------------------------
    def create_model(self, model_type="qsrcnn"):
        if model_type == "qsrcnn":
            frame_len = self.frame_len
            n1 = self.model_params['n1']
            n2 = self.model_params['n2']
            n3 = self.model_params['n3']

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
        elif model_type == "wavenet":
            frame_len = self.frame_len

            ae_width = 16
            ae_filter_length = 3

            ae_num_stages = 2
            ae_num_layers = 6

            num_stages = 2
            num_layers = 6

            width = 16
            skip_width = 16
            filter_length = 3

            input_sque = Input(shape=(frame_len, 1), name='input_layer')

            # ---------------------------------------
            # The Non-Causal Temporal Encoder.
            # ---------------------------------------
            en = Conv1D(ae_width, ae_filter_length, padding='same', name='ae_startconv')(input_sque)

            for num_layer in range(ae_num_layers):
                # dilation: 2**(0 1 2 3 4)
                d = Activation(selu)(en)
                d = Conv1D(ae_width, 3, padding='same', dilation_rate=2 ** (num_layer % ae_num_stages),
                           name='ae_dilatedconv_%d' % (num_layer + 1))(d)
                d = Activation(selu)(d)

                en2 = Conv1D(ae_width, 1, padding='same', dilation_rate=2 ** (num_layer % ae_num_stages),
                             name='ae_res_%d' % (num_layer + 1))(d)
                en = Add()([en2, en])

            en = Activation(selu)(en)
            en = Conv1D(16, 1, padding='causal', dilation_rate=1, name='ae_bottleneck')(en)
            en = Activation(selu)(en)
            en = AveragePooling1D(2, name='ae_pool')(en)
            # encoding = en

            # ---------------------------------------
            # The WaveNet Decoder.
            # ---------------------------------------
            # enup = UpSampling1D(2, name='up_sampling')(en)
            # l = shift_right(input_frame)

            l = Conv1D(width, filter_length, padding='causal', dilation_rate=1, name='startconv')(input_sque)
            l = Activation(selu)(l)
            # Set up skip connections.
            s = Conv1D(skip_width, 1, padding='causal', dilation_rate=1, name='skip_start')(l)
            s = Activation(selu)(s)

            # Residual blocks with skip connections.
            for i in range(num_layers):
                d = Conv1D(2 * width, filter_length, padding='causal', dilation_rate=2 ** (i % num_stages),
                           name='dilatedconv_%d' % (i + 1))(l)
                d = Activation(selu)(d)

                en3 = Conv1D(2 * width, 1, padding='causal', dilation_rate=1, name='cond_map_%d' % (i + 1))(en)  # 40
                en3 = Activation(selu)(en3)
                en3 = UpSampling1D(2, name='up_sampling_%d' % (i + 1))(en3)
                # d = condition(d,en3)
                d = Add()([d, en3])

                d_sigmoid = Activation('sigmoid')(d)
                d_tanh = Activation('tanh')(d)
                d = Multiply()([d_sigmoid, d_tanh])

                l2 = Conv1D(width, 1, padding='causal', dilation_rate=1, name='res_%d' % (i + 1))(d)
                l2 = Activation(selu)(l2)
                l = Add()([l2, l])

                s2 = Conv1D(skip_width, 1, padding='causal', dilation_rate=1, name='skip_%d' % (i + 1))(d)
                s = Add()([s2, s])

            s = Activation(selu)(s)

            s = Conv1D(skip_width, 3, padding='causal', activation='linear', name='output_layer1')(s)
            s = Activation(selu)(s)
            en4 = Conv1D(skip_width, 1, padding='causal', activation='linear', name='cond_map_out1')(en)
            en4 = Activation(selu)(en4)
            en4 = UpSampling1D(2, name='up_sampling')(en4)
            s = Add()([en4, s])
            s = Activation(selu)(s)

            outs = Conv1D(1, 3, padding='causal', activation='linear', name='output_layer')(s)

            model = Model(input_sque, outs)

        elif model_type == "autoencoder":
            frame_len = self.frame_len
            n1 = 64
            n2 = 32

            input_sque = Input(shape=(frame_len, 1))
            c1 = Conv1D(n1, 3, padding='same')(input_sque)
            c1 = Activation(selu)(c1)
            x = MaxPooling1D(2)(c1)

            c2 = Conv1D(n2, 3, padding='same')(x)
            c2 = Activation(selu)(c2)
            encoded = MaxPooling1D(2)(c2)

            d1 = UpSampling1D(2)(encoded)
            d1 = Conv1D(n2, 3, padding='same')(d1)
            d1 = Activation(selu)(d1)
            y = Activation(selu)(d1)

            d2 = UpSampling1D(2)(y)
            d2 = Conv1D(n1, 3, padding='same')(d2)
            d2 = Activation(selu)(d2)

            decoded = Conv1D(1, 5, padding='same', activation='linear')(d2)

            model = Model(input_sque, decoded)

        elif model_type == "esrcnn":
            f1 = 5
            f2_1 = 1
            f2_2 = 2
            f2_3 = 3
            f3 = 5

            n1 = 128
            n2 = 64

            frame_len = self.frame_len

            input_img = Input(shape=(frame_len, 1))
            x = Conv1D(n1, f1, padding='same', name='level1')(input_img)
            x = Activation(selu)(x)

            x1 = Conv1D(n2, f2_1, padding='same', name='lavel1_1')(x)
            x1 = Activation(selu)(x1)
            x2 = Conv1D(n2, f2_2, padding='same', name='lavel1_2')(x)
            x2 = Activation(selu)(x2)
            x3 = Conv1D(n2, f2_3, padding='same', name='lavel1_3')(x)
            x3 = Activation(selu)(x3)

            x = Average()([x1, x2, x3])

            out = Conv1D(1, f3, padding='same', activation='linear', name='output_1')(x)
            # out = LeakyReLU(0.2)(out)

            model = Model(input_img, out)
		"""
        elif model_type == "subpixel":
            frame_len = self.frame_len

            input_frame = Input(shape=(frame_len, 1))
            x = Conv1D(64, 5, padding='same', name='level1')(input_frame)
            x = Activation(selu)(x)
            x = Conv1D(32, 3, padding='same', name='level2')(x)
            x = Activation(selu)(x)

            x = self._upscale_block(x, 1)

            out = Conv1D(1, 5, activation='linear', padding='same', name='output_1')(x)

            model = Model(input_frame, out)
		"""
		
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

        drop = 0.5
        epochs_drop = 4.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lrate)
        lrate = K.get_value(self.model.optimizer.lr)
        print('Ir reduced from %f to %f' % (old_lr, lrate))
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
        reduce_LR = ReduceLROnPlateau(monitor='val_snr', factor=0.6, patience=2, verbose=1, mode='max', epsilon=0.0001, cooldown=0, min_lr=0)

        best_weights = self.weights_dir + '/' + self.codec + '_Type' + self.type + '_CepstralQSRCNN_Weights_Best_bs' + \
                       str(batch_size) + '_lr' + str(learning_rate) + '.h5'
        best_weights = os.path.normcase(best_weights)
        model_save = ModelCheckpoint(best_weights, monitor='val_snr', save_best_only=True, mode='max', save_weights_only=True, period=1)

        logger_name = self.log_dir + '/' + self.codec + '_Type' + self.type + '_CepstralQSRCNN_log_bs' + \
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
                              callbacks=[reduce_LR, stop_str, model_save, logger])

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
        TrainLossVec = trian_curve_root + '/' + self.codec + '_Type' + self.type + '_CepstralDDQSRCNN_TrainLoss_bs' + \
                       str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        TrainLossVec = os.path.normcase(TrainLossVec)

        sio.savemat(TrainLossVec, {'Train_Loss_Vec': hist.history['loss']})

        # ---------------------------------------------------------
        # 2. Saving Training Metric
        # ---------------------------------------------------------
        TrainSNRVec = trian_curve_root + '/' + self.codec + '_Type' + self.type + '_CepstralDDQSRCNN_TrainMetrice_bs' + \
                      str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        TrainSNRVec = os.path.normcase(TrainSNRVec)
        sio.savemat(TrainSNRVec, {'Train_SNR_Vec': hist.history['snr']})  # snr

        # ---------------------------------------------------------
        # 3. Saving Validation Loss
        # ---------------------------------------------------------
        ValiLossVec = trian_curve_root + '/' + self.codec + '_Type' + self.type + '_CepstralDDQSRCNN_ValiLoss_bs' + \
                      str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        ValiLossVec = os.path.normcase(ValiLossVec)
        sio.savemat(ValiLossVec, {'Vali_Loss_Vec': hist.history['val_loss']})

        # ---------------------------------------------------------
        # 4. Saving Validation Metric
        # ---------------------------------------------------------
        ValiSNRVec = trian_curve_root + '/' + self.codec + '_Type' + self.type + '_CepstralDDQSRCNN_ValiMetrice_bs' + \
                     str(self.batch_size) + '_lr' + str(self.learning_rate) + '.mat'
        ValiSNRVec = os.path.normcase(ValiSNRVec)
        sio.savemat(ValiSNRVec, {'Vali_SNR_Vec': hist.history['val_snr']})  # val_snr

        print("> Saving Completed, Time : ", time.time() - start)
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

    # -------------------------------------------------------------------------------
    # 4. Evaluate the Trained Model
    # -------------------------------------------------------------------------------
    def evaluation_model(self, x_test_noisy, detail_type="1", weights_path=""):
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

        preOutput = pre_file_root + "/" + self.codec + '_CNN_testplan_Type' + detail_type + "_ceps_vec.mat"
        preOutput = os.path.normcase(preOutput)

        sio.savemat(preOutput, {'predictions': predicted})
        print('> 3. Evaluation Result Saving Completed, Time : ' + str(time.time() - start))
        print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')