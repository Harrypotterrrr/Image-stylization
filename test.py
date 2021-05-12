import torch 

import numpy as np

from data_loader import dataset, data_loader
from common import *
from utility import ImageProcess
from model import TransformNet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

style_img = ImageProcess.read_image(style_img_path, uniform_h, uniform_w).to(device)

# test module
transform_net = TransformNet(32)

# load weights
transform_net.load_state_dict(torch.load(model_save_path))

#  to use GPU to test must follow the weight loading
transform_net.to(device)

transform_net.eval()

for param_tensor in transform_net.state_dict():
    print(param_tensor, "\t", transform_net.state_dict()[param_tensor].size())


# optimizer = torch.optim.SGD(transform_net.parameters(), lr=0.001, momentum=0.9)
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])


# print(transform_net)

test_indices = np.random.randint(0, len(data_loader), (test_num, ))
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler)

for i, (content_img, _) in enumerate(test_loader):

    """
    "Expected object of type torch.cuda.FloatTensor but found type torch.FloatTensor for argument #4 'mat1'": 
        while Module.to() is an in-place operator, Tensor.to() is not
    Reference: https://stackoverflow.com/questions/51605893/why-doesnt-my-simple-pytorch-network-work-on-gpu-device
    """
    content_img = content_img.to(device)
    output_img = transform_net(content_img)
    ImageProcess.save_paint_plot(style_img, content_img, output_img, "%stest_%d.jpg" % (output_img_path, i))
