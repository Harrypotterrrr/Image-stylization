import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):

    def __init__(self, features):
        """
        initialization
        :param features: the module container
        """
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_dict = {
            '3': "relu_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3",
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        :return: the specified feature of the output from intermediate layers
        """
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_dict:
                outputs.append(x)
        return outputs

