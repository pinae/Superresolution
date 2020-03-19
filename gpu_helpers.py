# -*- coding: utf-8 -*-
from tensorflow import config


def init_all_gpu():
    all_gpu = config.experimental.list_physical_devices('GPU')
    if all_gpu:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in all_gpu:
                config.experimental.set_memory_growth(gpu, True)
            logical_gpu_list = config.experimental.list_logical_devices('GPU')
            print(len(all_gpu), "Physical GPUs,", len(logical_gpu_list), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
