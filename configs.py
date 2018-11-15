# -*- coding: utf-8 -*-
#pylint: skip-file
import os

class CommonConfigs(object):
    def __init__(self, d_type):
        self.ROOT_PATH = ""
        self.TRAINING_DATA_PATH = self.ROOT_PATH + d_type + "/train_set/"
        self.VALIDATE_DATA_PATH = self.ROOT_PATH + d_type + "/validate_set/"
        self.TESTING_DATA_PATH = self.ROOT_PATH + d_type + "/test_set/"
        self.RESULT_PATH = self.ROOT_PATH + d_type + "/result/"
        self.MODEL_PATH = self.ROOT_PATH + d_type + "/model/"
        self.BEAM_SUMM_PATH = self.RESULT_PATH + "/beam_summary/"
        self.BEAM_GT_PATH = self.RESULT_PATH + "/beam_ground_truth/"
        self.GROUND_TRUTH_PATH = self.RESULT_PATH + "/ground_truth/"
        self.SUMM_PATH = self.RESULT_PATH + "/summary/"
        self.TMP_PATH = self.ROOT_PATH + d_type + "/tmp/"


class DeepmindTraining(object):
    def __init__(self, batch_size):
        self.IS_UNICODE = False
        self.REMOVES_PUNCTION = False
        self.HAS_Y = True
        self.BATCH_SIZE = batch_size

class DeepmindTesting(object):
    IS_UNICODE = False
    HAS_Y = True
    BATCH_SIZE = 100
    MIN_LEN_PREDICT = 35
    MAX_LEN_PREDICT = 120
    MAX_BYTE_PREDICT = None
    PRINT_SIZE = 500
    REMOVES_PUNCTION = False

class DeepmindConfigs():
    def __init__(self, colab):
        if colab:
            self.cc = CommonConfigs("drive/My Drive/IR2/deepmind/")
        else:
            self.cc = CommonConfigs("../deepmind/")

        self.CELL = "lstm" # gru or lstm
        self.CUDA = True
        self.COPY = True
        self.COVERAGE = True
        self.BI_RNN = True
        self.BEAM_SEARCH = True
        self.BEAM_SIZE = 4
        self.AVG_NLL = True
        self.NORM_CLIP = 2
        if not self.AVG_NLL:
            self.NORM_CLIP = 5
        self.LR = 0.15

        self.DIM_X = 128
        self.DIM_Y = self.DIM_X

        self.TF_P = 0.2

        self.MIN_LEN_X = 10
        self.MIN_LEN_Y = 10
        self.MAX_LEN_X = 400
        self.MAX_LEN_Y = 100
        self.MIN_NUM_X = 1
        self.MAX_NUM_X = 1
        self.MAX_NUM_Y = None

        self.NUM_Y = 1
        self.HIDDEN_SIZE = 256

        self.UNI_LOW_FREQ_THRESHOLD = 10

        self.PG_DICT_SIZE = 50000 # dict for acl17 paper: pointer-generator

        self.W_UNK = "<unk>"
        self.W_BOS = "<bos>"
        self.W_EOS = "<eos>"
        self.W_PAD = "<pad>"
        self.W_LS = "<s>"
        self.W_RS = "</s>"