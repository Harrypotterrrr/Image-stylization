import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as tv_models

import model
import utility

debug = True
iter_times = 1000
uniform_h = 200
uniform_w = 200
output_img_path = "./image/output_img.jpg"
style_img_path = "./image/style_img.jpg"
content_img_path = "./image/content_img.jpg"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# reference pytorch example: https://github.com/pytorch/examples/blob/0.4/fast_neural_style/neural_style/utils.py#L21-L26
def gram_matrix(x):
    """
    :param x: torch tensor
    :return: the gram matrix of x
    """
    (b, c, h, w) = x.size()
    phi = x.view(b, c, h * w)
    phi_T = phi.transpose(1, 2)
    return phi.bmm(phi_T) / (c * h * w) # use batch matrix(vector) inner product


# load image to torchTensor
style_img = utility.ImageProcess.read_image(style_img_path).to(device)
print("style image shape:", style_img.shape)

content_img = utility.ImageProcess.read_image(content_img_path).to(device)
print("content image shape:", content_img.shape)

# paint images
utility.ImageProcess.paint_image(style_img,"style_image")
utility.ImageProcess.paint_image(content_img,"content_image")

# build feature model
vgg16 = tv_models.vgg16(pretrained=True)
# convert into self feature extraction model
vgg16 = model.VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
if debug:
    print(vgg16)

# get features
style_features = vgg16(style_img)
content_features = vgg16(content_img)
if debug:
    print("style feature:")
    print([i.shape for i in style_features])
    print("content feature:")
    print([i.shape for i in content_features])

# calculate Gram matrix according to the extracted feature
style_gram = [gram_matrix(i) for i in style_features]
if debug:
    print("style Gram matrix:")
    print([i.shape for i in style_gram])

# the stage of train the model
## get the copy of input image as input and set its parameters able to be trained
output_img = content_img.clone().requires_grad_(True)
optimizer = optim.LBFGS([output_img])

## set hyperparameter
style_weight = 1e7
content_weight = 0.5

## build an list item to be a counter of the closure
it = [0]

## train the model
while it[0] < iter_times:

    def closure():
        optimizer.zero_grad() # TODO
        output_features = vgg16(output_img)

        # summarize the loss between output_img and style_img, content_img
        content_loss = F.mse_loss(input=output_features[2], target=content_features[2])
        style_loss = 0
        output_gram = [gram_matrix(i) for i in output_features]
        for og, sg in zip(output_gram, style_gram):
            style_loss += F.mse_loss(input=og, target=sg)
        # factors of the tradeoff between style_loss and content_loss is hyperparameters
        loss = style_loss * style_weight + content_loss * content_weight

        if it[0] % 20 == 0:
            print("Step %d: style_loss: %.5f content_loss: %.5f" % (it[0], style_loss, content_loss))
        if it[0] % 100 == 0:
            utility.ImageProcess.paint_image(output_img, title='Output Image')

        # calculate gradient through backward
        loss.backward()
        it[0] += 1

        return loss

    # LBFGS optimizer to update parameters needs a closure that reevaluates the model and returns the loss
    optimizer.step(closure)

utility.ImageProcess.paint_image(output_img, title='Output Image')
utility.ImageProcess.save_image(output_img, output_img_path)