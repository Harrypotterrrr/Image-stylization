import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as tv_models

from tqdm import tqdm

from model import VGG, TransformNet
from utility import gram_matrix, ImageProcess
from data_loader import data_loader

debug = True
iter_times = 1000
uniform_h = 200
uniform_w = 200

output_img_path = "./image/"
style_img_path = "./image/style_img.jpg"
model_save_path = "./model.pkl"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# load image to torchTensor
style_img = ImageProcess.read_image(style_img_path).to(device)
print("style image shape:", style_img.shape)

# paint images
if debug:
    ImageProcess.paint_image(style_img,"style_image")

# build feature model
vgg16 = tv_models.vgg16(pretrained=True)
# convert into self feature extraction model
vgg16 = VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
if debug:
    print(vgg16)

# get style image features
style_features = vgg16(style_img)
if debug:
    print("style feature:")
    print([i.shape for i in style_features])

# calculate Gram matrix according to the extracted feature
style_gram = [gram_matrix(i) for i in style_features]
if debug:
    print("style Gram matrix:")
    print([i.shape for i in style_gram])



transform_net = TransformNet(32).to(device)

verbose_batch = 100

## set hyperparameter
epoch_times = 1
style_weight = 1e5
content_weight = 1
totalVariation_weight = 1e-6

optimizer = optim.Adam(transform_net.parameters(), lr=1e-3)
transform_net.train() ## Sets the module parameter in training mode.


for epoch in range(epoch_times):
    print("Epoch %d" % epoch)

    with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
        for batch, (content_images, _) in pbar:
            optimizer.zero_grad()

            content_images = content_images.to(device)
            output_images = transform_net(content_images)
            output_images = output_images.clamp(-4, 4) # TODO

            content_features = vgg16(content_images)
            output_features = vgg16(output_images)

            # content loss
            content_loss = F.mse_loss(output_features[1], content_features[1]) # TODO [1]

            # total variation loss
            ##pass

            # style loss
            style_loss = 0
            output_gram = [gram_matrix(x) for x in output_features]
            for og, sg in zip(output_gram, style_gram):
                style_loss += F.mse_loss(og, sg.expand_as(og))

            loss = content_weight * content_loss + style_weight * style_loss

            loss.backward()
            optimizer.step()

            sstr = "Step %d: style_loss: %.5f content_loss: %.5f" % (batch, style_loss, content_loss)
            if batch % verbose_batch == 0:
                s = '\n' + s
                ImageProcess.save_image(output_images,"%s_%d.jpg" %(output_img_path,batch))
            pbar.set_description(sstr)

torch.save(transform_net.state_dict(), 'transform_net.pth') # TODO
torch.save(transform_net.state_dict(), model_save_path)


"""
# the stage of train the model



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



"""

