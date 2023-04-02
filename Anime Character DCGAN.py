# import the required packages
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module): 

    def __init__(self):
        super(G, self).__init__() # We inherit from the nn.Module tools.
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # We start with an inversed convolution.
            nn.BatchNorm2d(512), # We normalize all the features along the dimension of the batch.
            nn.ReLU(True), # We apply a ReLU rectification to break the linearity.
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128),
            nn.ReLU(True), 
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
        )

    def forward(self, input):
        output = self.main(input) 
        return output 



class D(nn.Module): # We introduce a class to define the discriminator.

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), # We start with a convolution.
            nn.LeakyReLU(0.2, inplace = True), # We apply a LeakyReLU.
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
        )

    def forward(self, input): 
        output = self.main(input) 
        return output.view(-1)


if __name__ == '__main__':
    # Setting some hyperparameters
    batchSize = 64 # We set the size of the batch.
    imageSize = 64 # We set the size of the generated images (64x64).

    # Creating the transformations
    transform = transforms.Compose([transforms.Resize((imageSize,imageSize)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 

    # Loading the dataset
    dataset = datasets.ImageFolder(root = './data',  transform = transform) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 

    # Creating the generator
    netG = G() # We create the generator object.
    netG.apply(weights_init) # We initialize all the weights of its neural network.

    # Creating the discriminator
    netD = D() # We create the discriminator object.
    netD.apply(weights_init) # We initialize all the weights of its neural network.


    criterion = nn.BCELoss() # We create a criterion object that will measure the error between the prediction and the target.
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    for epoch in range(50): 
        print('epoch :',epoch)
        for i, data in enumerate(dataloader, 0):
            print('Image dataset :',i)

            # 1st Step: Updating the weights of the neural network of the discriminator

            netD.zero_grad() # We initialize to 0 the gradients of the discriminator with respect to the weights.
            print('1st Step: We initialize to 0 the gradients of the discriminator with respect to the weights')
            
            # Training the discriminator with a real image of the dataset
            real, _ = data
            input = Variable(real)
            target = Variable(torch.ones(input.size()[0]))
            output = netD(input)
            errD_real = criterion(output, target)
            print('Training the discriminator with a real image of the dataset')

            # Training the discriminator with a fake image generated by the generator
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0])) 
            output = netD(fake.detach())
            errD_fake = criterion(output, target)
            print('Training the discriminator with a fake image generated by the generator')
            
            # Backpropagating the total error
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            print('Backpropagating the total error')

            # 2nd Step: Updating the weights of the neural network of the generator

            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0])) # We get the target.
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()
            print('2nd Step: Updating the weights of the neural network of the generator\n')
            

            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

            if i % 100 == 0: # Every 100 steps:
                vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
                fake = netG(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
                print('3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps\n')


