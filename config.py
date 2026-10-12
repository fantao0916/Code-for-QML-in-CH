import os
import torch
import json
from torch.utils.data.dataset import Dataset
#from transformers import BertTokenizer


class CommonConfig:
    output_dir = "./checkpoint/"


class Config:
    def __init__(self, data_name):
        cf = CommonConfig()
        self.output_dir = cf.output_dir
        self.output_dir = os.path.join(self.output_dir, data_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_dir = cf.data_dir

        self.epochs = 1
        self.train_batch_size = 64
        self.dev_batch_size = 64
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.01
        self.save_step = 500

        # quantum parts;
        self.qubits = 6
        self.qlayers = [7]
        self.enc = "amp"
