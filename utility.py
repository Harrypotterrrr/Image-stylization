import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

norm_mean = [0.485, 0.456, 0.406] # convention
norm_std = [0.229, 0.224, 0.225]

class ImageProcess():

    @staticmethod
    def torchTensor2Image(tensor):
        """
        convert torch tensor range in [0,1] of shape(B*C*H*W) to numpy array range in [0,255] of shape(B*H*W*C)
        :param tensor: torch tensor
        :return: numpy array
        """
        image = tensor.detach().cpu().numpy()
        image = image * np.array(norm_std).reshape((1, 3, 1, 1)) + np.array(norm_mean).reshape((1, 3, 1, 1))
        image = image.transpose(0, 2, 3, 1) * 255.
        # clip and change type to integer
        image = image.clip(0, 255).astype(np.uint8)

        return image[0]

    @staticmethod
    def preprocess_image(img, h = None, w = None):
        """
        preprocess an image
        :param img: PIL image
        :param h: height
        :param w: width
        :return: FloatTensor with 4 dimension(B, C, H, W)
        """
        norm = transforms.Normalize(mean=norm_mean, std=norm_std)

        if h and w:
            t = transforms.Compose([
                transforms.Resize((h, w)),
                transforms.CenterCrop((h, w)),
                transforms.ToTensor(), # squash PIL image in range[0,255] of shape(H*W*C) to a FloatTensor in range[0,1] of shape(C*H*W)
                norm,
            ])
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                norm,
            ])
        return t(img).unsqueeze(0)

    @staticmethod
    def read_image(path, h = None, w = None):
        """
        open an image
        :param path: the path of image
        :return: torch tensor of an image
        """
        img = Image.open(path)
        tensor_img = ImageProcess.preprocess_image(img, h, w)
        return tensor_img

    @staticmethod
    def paint_image(tensor, title=None):
        """
        paint the image after recover it to numpy array
        :param ts: torch tensor of image
        :return: NULL
        """
        image = ImageProcess.torchTensor2Image(tensor)
        print(image.shape)
        plt.axis('off')
        plt.imshow(image)
        plt.show()
        if title is not None:
            plt.title(title)

    @staticmethod
    def save_image(tensor, path):
        """
        save the image converted from a tensor fromat
        :param tensor: torch tensor of the image
        :param path: oriented path to save the image
        """
        img = ImageProcess.torchTensor2Image(tensor)
        Image.fromarray(img).save(path)
        print("Successfully save the final stylized image to:", path)
