# DCGAN in PyTorch
<img src="https://user-images.githubusercontent.com/50144683/229373082-03d5f09d-0c07-48a5-9c67-fc1ee87057da.gif" width=55% height=45%></br>
**Picture:** _These anime faces were produced by our generator from 1 to 50 epochs on different aspects of the images._

This repository contains the Pytorch implementation of the following paper:
>**Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**</br>
>Alec Radford, Luke Metz, Soumith Chintala</br>
>https://arxiv.org/abs/1511.06434
>
>**Abstract:** _In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations._

## Architecture
<img src="https://user-images.githubusercontent.com/50144683/229377208-6c607383-7b43-4a95-a8b3-59a59db6697d.png" width=75%  height=20%></br>
DCGAN uses convolutional and convolutional-transpose layers in the generator and discriminator, respectively. Here the discriminator consists of strided convolution layers, batch normalization layers, and LeakyRelu as activation function. It takes a 3x64x64 input image. The generator consists of convolutional-transpose layers, batch normalization layers, and ReLU activations. The output will be a 3x64x64 RGB image.

## Usage
First, download the Anime Faces dataset from [Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset)
+ Create folders 'data' and 'results'
+ Place the 'Images' folder downloaded from Kaggle in the 'data' folder.
+ Run the Python file
+ For every 100<sup>th</sup> iteration in every Epoch, the fake image generated by the Generator will be saved in the 'results' folder. In total, we have 50 fake images.

## Results
After 1<sup>st</sup> Epoch:</br>
<img src="https://user-images.githubusercontent.com/50144683/229377671-29abef1a-c44c-4995-9ff3-606f81ebe48f.png" width=55% height=45%></br>
After 50<sup>th</sup> Epoch:</br>
<img src="https://user-images.githubusercontent.com/50144683/229378004-cbe0a381-3538-48cb-b060-be1768be2583.png" width=55% height=45%></br>
More results can be found [here](https://github.com/arunsandy1309/Anime-Character-Generation-DCGAN/tree/main/results)

## Related Works
+ [Vanilla GAN in PyTorch](https://github.com/arunsandy1309/Vanilla-GAN)
+ [Generating Real World Images using DCGAN in PyTorch](https://github.com/arunsandy1309/RealWorld-Image-Generation-DCGAN)
+ [Conditional GAN in PyTorch](https://github.com/arunsandy1309/Conditional-GAN-PyTorch)
+ [CycleGAN in PyTorch](https://github.com/arunsandy1309/CycleGAN-PyTorch)
