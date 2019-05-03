import torch
import torchvision
from torchvision import transforms

from utility import ImageProcess

batch_size = 4
height = 256
width = 256

# load transformation function
data_transform = ImageProcess.common_process(height, width)


dataset = torchvision.datasets.ImageFolder('/home/jiahl/coco/', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(dataset)