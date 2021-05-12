import torch
import torchvision

from utility import ImageProcess
from common import *


# load transformation function
data_transform = ImageProcess.common_transforms(uniform_h, uniform_w)

#dataset = torchvision.datasets.CocoDetection(root='../train2014', annFile='../annotations/instances_train2014.json', transforms=data_transform)
dataset = torchvision.datasets.CIFAR10(root='CIFAR10', transform=data_transform, download=True)
#dataset = torchvision.datasets.ImageFolder('/home/jiahl/coco/', transform=data_transform)
# dataset = torch.utils.data.Subset(dataset, [i for i in range(100)])
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if verbose_print:
    print("dataset description:\n", dataset)

if __name__ == "__main__":
    print(type(dataset))
