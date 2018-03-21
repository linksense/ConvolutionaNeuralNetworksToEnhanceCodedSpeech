#########################################################################################################
# WaveformQSRCNN_TrainTest_GPUs.py: Train and Test QSRCNN for G711/ADPCM/AMR/EVS using Waveform features
# Author:   Huijun Liu
# Time:     10.05.2017
# Location: TU Braunschweig IfN
#########################################################################################################

import os
import sys
import time

import WaveformQSRCNN as model
import tensorflow as tf
import DataPrepare as dp
from keras.backend.tensorflow_backend import set_session

#####################################################################################
# 0. Setteings For GPUs and Parameters
#####################################################################################
using_gpu = 0
if using_gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"                  # x stand for GPU index: 3-x!!
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # Only 30% Memory of GPUs can be used
    set_session(tf.Session(config=config))

train_or_test = "train"  # train  or  test

default_opt_params = {'lr': 5e-4, 'batch_size': 32, 'nb_epochs': 1000}
default_model_params = {'n1': 32, 'n2': 64, 'n3': 32, 'frame_len': 80}
codec_type_params = {'weights_dir': "./model_weights", 'logdir': "./log"}

train_inputs = ""
train_targets = ""

vali_inputs = ""
vali_targets = ""

test_inputs = ""

if train_or_test == "train":
    # -------------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------------
    train_inputs = "./TrainValiData/Train_inputSet_ALaw_defautLang_OLdata_ValiTrain_Reverse_Extended_Reverse.mat"
    train_targets = "./TrainValiData/Train_targetSet_ALaw_defautLang_OLdata_ValiTrain_Reverse_Extended_Reverse.mat"

    vali_inputs = "./TrainValiData/Vali_inputSet_ALaw_defautLang_OLdata_ValiTrain_smallVali_Reverse_Extended.mat"
    vali_targets = "./TrainValiData/Vali_targetSet_ALaw_defautLang_OLdata_ValiTrain_smallVali_Reverse_Extended.mat"

    x_train_noisy, x_train, x_train_noisy_vali, x_train_vali = dp.load_train_data(train_inputs, train_targets,
                                                                                  vali_inputs, vali_targets)

    # -------------------------------------------------------------------------------
    # 2. Init Cepstral-QSRCNN Model
    # -------------------------------------------------------------------------------
    qsrcnn = model.WaveformQSRCNN(opt_params=default_opt_params,
                                  model_params=default_model_params,
                                  codec_type_params=codec_type_params)

    # -------------------------------------------------------------------------------
    # 3. Fit The Cepstral-QSRCNNe Model
    # -------------------------------------------------------------------------------
    hist =qsrcnn.fit(x_train_noisy, x_train, x_train_noisy_vali, x_train_vali)

    # -------------------------------------------------------------------------------
    # 4. Save Weights and Traning Curves
    # -------------------------------------------------------------------------------
    qsrcnn.save_weights()
    qsrcnn.save_training_curves(hist=hist)

elif train_or_test == "test":
    # -------------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------------
    test_inputs = "inputTestSet_g711concat_PDandOLAI_Frame_80v73.mat"
    x_test_noisy = dp.load_test_data(test_inputs)

    # -------------------------------------------------------------------------------
    # 2. Init Cepstral-QSRCNN Model
    # -------------------------------------------------------------------------------

    codec_type_params = {'weights_dir': "./model_weights", 'logdir': "./log"}
    qsrcnn = model.WaveformQSRCNN(opt_params=default_opt_params,
                                  model_params=default_model_params,
                                  codec_type_params=codec_type_params)

    # -------------------------------------------------------------------------------
    # 3. Evaluate The Cepstral-QSRCNNe Model
    # -------------------------------------------------------------------------------
    qsrcnn.evaluation_model(x_test_noisy)

else:
    raise Exception("Do you want to  train or test the model ? Please set the variable train_or_test !")
    # sys.exit("Please set the codec name !")
