import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def set_gpu(device):

    if device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.75
        config.gpu_options.allow_growth = True
        #session = InteractiveSession(config=config)
        InteractiveSession(config=config)

    elif device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
