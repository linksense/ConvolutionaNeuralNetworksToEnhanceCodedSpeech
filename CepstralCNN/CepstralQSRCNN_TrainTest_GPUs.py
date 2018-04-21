#######################################################################################################################
# Project QSRCNN: Quantized Speech Reconstruction using Convolutional Neural Network     V1.0
# =====================================================================================================================
# CepstralQSRCNN_TrainTest_GPUs.py: Train and Test QSRCNN for G711/ADPCM/AMR/EVS using Cepstral features
#
#
# =====================================================================================================================
# Technische Universitaet Braunschweig, IfN
# Author:    Huijun Liu M.Sc.
# Date:      17.06.2017
#######################################################################################################################

import os
import sys
import time

import CepstralQSRCNN as model
import tensorflow as tf
import DataPrepare as dp
from keras.backend.tensorflow_backend import set_session

#####################################################################################
# 0. Settings For GPU
#####################################################################################
using_gpu = 0
if using_gpu == 1:                                               # Only one GPU can be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"                     # x stand for GPU index: 3-x!!
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4     # Up to 80% Memory of GPUs can be used
    set_session(tf.Session(config=config))

#####################################################################################
# 1. Settings Parameters
#####################################################################################

train_or_test = "test"            # train  or  test  the deep model
codec = "amrwb"                    # g711/adpcm/amrwb/evsswb  codec can be used
type = "3"                      # 1_2  or  3             for Training
type_detail = "3"                 # 1  or  2  or  3 or 4   for Testing
frame_len = ""                    # 256(g711/adpcm)  or  512(amrwb)  or  1024(evsswb)

if codec == "g711" or codec == "adpcm":
    default_model_params = {'n1': 16, 'n2': 32, 'n3': 16, 'frame_len': 32}        # Parameters for model itself
    frame_len = "256"  # 256(g711/adpcm)  or  512(amrwb)  or  1024(evsswb)        # Frame length
elif codec == "amrwb":
    default_model_params = {'n1': 32, 'n2': 64, 'n3': 32, 'frame_len': 64}
    frame_len = "512"  # 256(g711/adpcm)  or  512(amrwb)  or  1024(evsswb)
elif codec == "evsswb":
    default_model_params = {'n1': 64, 'n2': 128, 'n3': 64, 'frame_len': 128}
    frame_len = "1024"  # 256(g711/adpcm)  or  512(amrwb)  or  1024(evsswb)
else:
    raise Exception("Please set the variable codec !")

default_opt_params = {'lr': 5e-4, 'batch_size': 16, 'nb_epochs': 1000}            # Parameters for model training
codec_type_params = {'codec': codec, 'type': type,
                     'weights_dir': "./model_weights",
                     'logdir': "./log"}                                           # Other parameters

#####################################################################################
# 2. Training and Testing
#####################################################################################
train_inputs = ""          # Path of the input data for training
train_targets = ""         # Path of the target data for training

vali_inputs = ""           # Path of the input data for validation
vali_targets = ""          # Path of the target data for validation

test_inputs = ""           # Path of the input data for testing

if train_or_test == "train":
    # -------------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------------
    train_inputs = "./TrainValiData/Train_inputSet_" + codec + \
                   "_defautLang_OLdata_ValiTrain_type" + type + "_Fram256_ceps.mat"
    train_targets = "./TrainValiData/Train_targetSet_" + codec + \
                    "_defautLang_OLdata_ValiTrain_type" + type + "_Fram256_ceps.mat"

    vali_inputs = "./TrainValiData/Vali_inputSet_" + codec + \
                  "_defautLang_OLdata_ValiTrain_smallVali_type" + type + "_Fram256_ceps.mat"
    vali_targets = "./TrainValiData/Vali_targetSet_" + codec + \
                   "_defautLang_OLdata_ValiTrain_smallVali_type" + type + "_Fram256_ceps.mat"

    x_train_noisy, x_train, x_train_noisy_vali, x_train_vali = dp.load_train_data(train_inputs, train_targets,
                                                                                  vali_inputs, vali_targets)

    # -------------------------------------------------------------------------------
    # 2. Init Cepstral-QSRCNN Model
    # -------------------------------------------------------------------------------
    qsrcnn = model.CepstralQSRCNN(opt_params=default_opt_params,
                                  model_params=default_model_params,
                                  codec_type_params=codec_type_params)

    # -------------------------------------------------------------------------------
    # 3. Fit The Cepstral-QSRCNNe Model
    # -------------------------------------------------------------------------------
    hist =qsrcnn.fit(x_train_noisy, x_train, x_train_noisy_vali, x_train_vali)

    # -------------------------------------------------------------------------------
    # 4. Save Weights and Training Curves
    # -------------------------------------------------------------------------------
    qsrcnn.save_weights()
    qsrcnn.save_training_curves(hist=hist)

elif train_or_test == "test":
    # -------------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------------
    test_inputs = "inputTestSet_" + codec + "_concat_Type" + type_detail + "_Frame_" + frame_len + "_ceps_v73.mat"
    x_test_noisy = dp.load_test_data(test_inputs)

    # -------------------------------------------------------------------------------
    # 2. Init Cepstral-QSRCNN Model
    # -------------------------------------------------------------------------------
    if type_detail == "1" or type_detail == "2":
        type = "1_2"
    elif type_detail == "3" or type_detail == "4":
        type = "3"

    codec_type_params = {'codec': codec, 'type': type, 'weights_dir': "./model_weights", 'logdir': "./log"}
    qsrcnn = model.CepstralQSRCNN(opt_params=default_opt_params,
                                  model_params=default_model_params,
                                  codec_type_params=codec_type_params)

    # -------------------------------------------------------------------------------
    # 3. Evaluate The Cepstral-QSRCNNe Model
    # -------------------------------------------------------------------------------
    qsrcnn.evaluation_model(x_test_noisy, type_detail)

else:
    raise Exception("Do you want to  train or test the model ? Please set the variable train_or_test !")
    # sys.exit("Please set the codec name !")