#######################################################################################################################
# Project QSRCNN: Quantized Speech Reconstruction using Convolutional Neural Network     V1.0
# =====================================================================================================================
# DataPrepare.py: Data prepare and load data
#
#
# =====================================================================================================================
# Technische Universität Braunschweig, IfN
# Author:    Huijun Liu M.Sc.
# Date:      20.05.2017
#######################################################################################################################

import os
import time
import h5py as h5
import numpy as np
import scipy.io as sio

from numpy import random

# -------------------------------------------------------------------------------
# 1. load data
# -------------------------------------------------------------------------------


def load_train_data(train_inputs, train_targets, vali_inputs, vali_targets):
    print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
    print('> Loading data ')

    start = time.time()
    # ---------------------------------------------------------
    # 1. Load Input Data for Training
    # ---------------------------------------------------------
    mat_input = train_inputs
    mat_input = os.path.normcase(mat_input)
    print('> 1. Loading Training Input: ' + mat_input + '...')

    x_train_noisy = sio.loadmat(mat_input)
    x_train_noisy = x_train_noisy['inputSetNorm']
    x_train_noisy = np.array(x_train_noisy)

    # ---------------------------------------------------------
    # 2. Load Target Data for training
    # ---------------------------------------------------------
    mat_target = train_targets
    mat_target = os.path.normcase(mat_target)
    print('> 2. Loading Training Target: ' + mat_target + '...')

    x_train = sio.loadmat(mat_target)
    x_train = x_train['targetSet']
    x_train = np.array(x_train)
    # x_train = target_min_max_scaler.fit_transform(x_train)

    # ---------------------------------------------------------
    # 3. Load Input Data for Validation
    # ---------------------------------------------------------
    mat_input_vali = vali_inputs
    mat_input_vali = os.path.normcase(mat_input_vali)
    print('> 3. Loading Validation Input: ' + mat_input_vali + '...')

    x_train_noisy_vali = sio.loadmat(mat_input_vali)
    x_train_noisy_vali = x_train_noisy_vali['inputSetNorm']
    x_train_noisy_vali = np.array(x_train_noisy_vali)

    # ---------------------------------------------------------
    # 4. Load Target Data for Validation
    # ---------------------------------------------------------
    mat_target_vali = vali_targets
    mat_target_vali = os.path.normcase(mat_target_vali)
    print('> 4. Loading Validation Target: ' + mat_target_vali + '...')

    x_train_vali = sio.loadmat(mat_target_vali)
    x_train_vali = x_train_vali['targetSet']
    x_train_vali = np.array(x_train_vali)

    # ---------------------------------------------------------
    # 5. Randomization of Training and/or validation Pairs
    # ---------------------------------------------------------
    print('> 5. Randomization of Training Pairs ...')
    frame_length = x_train_noisy.shape[1]

    random.seed(1234)
    train = np.column_stack((x_train_noisy, x_train))
    np.random.shuffle(train)
    x_train_noisy = train[:, :frame_length]
    x_train = train[:, frame_length:]

    # validation = np.column_stack((x_train_noisy_vali, x_train_vali))
    # np.random.shuffle(validation )
    # x_train_noisy_vali = validation [:, :frame_length]
    # x_train_vali = validation [:, frame_length:]

    # ---------------------------------------------------------
    # 6. Reshape of Training and validation Pairs
    # ---------------------------------------------------------
    x_train_noisy = np.reshape(x_train_noisy, (x_train_noisy.shape[0], x_train_noisy.shape[1], 1))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_train_noisy_vali = np.reshape(x_train_noisy_vali, (x_train_noisy_vali.shape[0], x_train_noisy_vali.shape[1], 1))
    x_train_vali = np.reshape(x_train_vali, (x_train_vali.shape[0], x_train_vali.shape[1], 1))

    print("> Data Loaded, , Time : ", time.time() - start)
    print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

    return x_train_noisy, x_train, x_train_noisy_vali, x_train_vali


def load_test_data(testfile_path="inputTestSet_g711concat_Type3_Frame_256_ceps_v73.mat"):
    print('> Loading Test data... ')

    test_file_root = "./TestData"
    if not (os.path.exists(test_file_root)):
        os.makedirs(test_file_root)

    mat_input = test_file_root + "/" + testfile_path
    mat_input = os.path.normcase(mat_input)

    x_test_noisy = h5.File(mat_input, 'r')
    x_test_noisy = x_test_noisy.get('inputTestNorm')
    x_test_noisy = np.array(x_test_noisy)
    x_test_noisy = np.transpose(x_test_noisy)

    # x_test_noisy = sio.loadmat(mat_input)
    # x_test_noisy = x_test_noisy['inputTestNorm']
    # x_test_noisy = np.array(x_test_noisy)

    x_test_noisy = np.reshape(x_test_noisy,(x_test_noisy.shape[0], x_test_noisy.shape[1], 1))

    return x_test_noisy