#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 07:03:15 2021

@author: hexuerui
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as tv_models

import numpy as np
from tqdm import tqdm

from model import VGG, TransformNet
from utility import gram_matrix, total_variance, ImageProcess

from common import *

def style_distance_gram(product, style):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build feature model
    vgg16 = tv_models.vgg16(pretrained=True)
    # convert into self feature extraction model
    vgg16 = VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
    
    # get style image features
    style_features = vgg16(style)
    
    # calculate Gram matrix according to the extracted feature
    style_gram = [gram_matrix(i) for i in style_features]
    
    # get product features and Gram matrix
    product_features = vgg16(product)
    product_gram = [gram_matrix(x) for x in product_features]
    
    # calculate style distance
    style_distance = 0
    for pg, sg in zip(product_gram, style_gram):
        style_distance += F.mse_loss(pg, sg.expand_as(pg))
        
    return style_distance

def style_distance(product, style):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build feature model
    vgg16 = tv_models.vgg16(pretrained=True)
    # convert into self feature extraction model
    vgg16 = VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
    
    # get style image features
    style_features = vgg16(style)
    
    # get product features
    product_features = vgg16(product)
    
    return F.mse_loss(product_features[1], style_features[1])

def content_distance_gram(product, content):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build feature model
    vgg16 = tv_models.vgg16(pretrained=True)
    # convert into self feature extraction model
    vgg16 = VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
    
    content_features = vgg16(content)
    content_gram = [gram_matrix(x) for x in content_features]
    product_features = vgg16(product)
    product_gram = [gram_matrix(x) for x in product_features]
    
    content_distance = 0
    for pg, cg in zip(product_gram, content_gram):
        content_distance += F.mse_loss(pg, cg.expand_as(pg))

    return content_distance

def content_distance(product, content):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build feature model
    vgg16 = tv_models.vgg16(pretrained=True)
    # convert into self feature extraction model
    vgg16 = VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
    
    # get style image features
    content_features = vgg16(content)
    
    # get product features
    product_features = vgg16(product)
    
    return F.mse_loss(product_features[1], content_features[1])

def cosine_similarity(product, source):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build feature model
    vgg16 = tv_models.vgg16(pretrained=True)
    # convert into self feature extraction model
    vgg16 = VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
    
    # get source features
    source_features = vgg16(source)
    
    # get product features
    product_features = vgg16(product)
    
    csim = []
    avg = 0
    for pf, sf in zip(product_features, source_features):
        pf = pf.view(1, -1)
        sf = sf.view(1, -1)
        v = torch.cosine_similarity(pf, sf, dim=1)
        csim.append(v)
        #avg += v
    return csim, #avg/len(csim)

def show_distance(product, content, style):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    style_img = ImageProcess.read_image(style, uniform_h, uniform_w).to(device)
    content_img = ImageProcess.read_image(content, uniform_h, uniform_w).to(device)
    product_img = ImageProcess.read_image(product, uniform_h, uniform_w).to(device)
    print('gram style distance:', style_distance_gram(product_img, style_img))
    print('mse style distance:', style_distance(product_img, style_img))
    print('gram content distance:', content_distance_gram(product_img, content_img))
    print('mse content distance:', content_distance(product_img, content_img))
    style_csim = cosine_similarity(product_img, style_img)
    print('style cosine similarity on different levels', style_csim)
    #print('avg style cosine similarity:', style_avg)
    content_csim = cosine_similarity(product_img, content_img)
    print('content cosine similarity on different levels', content_csim)
    #print('avg style cosine similarity:', content_avg)
    if uniform_h != 299 or uniform_w != 299:
        style_img = ImageProcess.read_image(style, 299, 299).to(device)
        content_img = ImageProcess.read_image(content, 299, 299).to(device)
        product_img = ImageProcess.read_image(product, 299, 299).to(device)
    print('product-style FID:', FID(product_img, style_img))
    print('product-content IFD:', FID(product_img, content_img))
    
from scipy import linalg
def FID(src1, src2):
    # input: 2 lists processed images
    # input shape: batch*3*299*299
    model = tv_models.inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    
    src1 = np.array(src1)
    src1 = torch.from_numpy(src1)
    f1 = model(src1).detach().numpy()
    mu1 = np.mean(f1, axis=0)
    sigma1 = np.cov(f1, rowvar=False)
    
    src2 = np.array(src2)
    src2 = torch.from_numpy(src2)
    f2 = model(src2).detach().numpy()
    mu2 = np.mean(f2, axis=0)
    sigma2 = np.cov(f2, rowvar=False)
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    diff = mu1 - mu2
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert(mu1.shape == mu2.shape)
    assert(sigma1.shape == sigma2.shape)
    
    eps = 1e-4
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='input product/content/style file')
    parser.add_argument('product', help='product image')
    parser.add_argument('content', help='content image')
    parser.add_argument('style', help='style image')
    args = parser.parse_args()
    print(args)
    show_distance(args.product, args.content, args.style)
    
'''
/Users/hexuerui/Desktop/introDL/project/PytorchWCT-master/images/style/in4.jpg
/Users/hexuerui/Desktop/introDL/project/PytorchWCT-master/images/content/in2.jpg
/Users/hexuerui/Desktop/introDL/project/deep-transfer-master/outputs/trial_face_stylized_by_in4_alpha_20.jpg

/Users/hexuerui/Desktop/introDL/project/deep-transfer-master/inputs/contents/plane.jpeg
/Users/hexuerui/Desktop/introDL/project/PytorchWCT-master/images/style/in4.jpg
/Users/hexuerui/Desktop/introDL/project/deep-transfer-master/outputs/plane_stylized_by_in4_alpha_20.jpeg

'''

    