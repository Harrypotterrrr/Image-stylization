import torch

import numpy as np

from data_loader import dataset, data_loader
from common import *
from utility import ImageProcess
from model import TransformNet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

style_img = ImageProcess.read_image(style_img_path, uniform_h, uniform_w).to(device)

# test module
transform_net = TransformNet(32).to(device)

# load weights
transform_net.load_state_dict(torch.load(model_save_path))

# print(transform_net)

test_indices = np.random.randint(0, len(data_loader), (test_num, ))
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler)

for i, (content_img, _) in enumerate(test_loader):

    output_img = transform_net(content_img)
    ImageProcess.save_paint_plot(style_img, content_img, output_img, "%stest_%d.jpg" % (output_img_path, i))