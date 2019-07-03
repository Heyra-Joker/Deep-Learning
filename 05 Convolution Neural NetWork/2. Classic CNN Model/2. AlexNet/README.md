<figure class="third">
    <img src="../../../picture/heyra.png" width="50" heigth="50"/>
</figure>

# AlexNet

You need to view the original paper of [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

![](../../../picture/180.png)

In AlexNet, have three implementations of methods:

You can view [AlexNet(Theory)](1. AlexNet(Theory).ipynb)  to understanding AlexNet.

**Tensorflow**:

The code in this directory  [AlexNet-Tensorflow](2. AlexNet-Tensorflow)
This directory has three directories:

[1]  [TFRecord](2. AlexNet-Tensorflow/1. TFrecord)
this cell tells what is TFRecord and tell you how to build Google cloud:

> ![](../../../picture/197.png)

[2] [Load Data set](2. AlexNet-Tensorflow/2. LoadCatVsDogs)
this cell tells you how to load a larger data set of Tensorflow.
In this case, data using Kaggle-CatsVsDogs.

> ![](fils/01.png)

[3] [AlexNet-CatVsDogs-Model](2. AlexNet-Tensorflow/3. AlexNet-CatVsDogs-Model)

If you are just careful about the AalexNet model, you can view this directory.

> ![](2. AlexNet-Tensorflow/3. AlexNet-CatVsDogs-Model/alexnet.png)



**Pytorch**:
The code in this directory [AlexNet-Pytorch](3. AlexNet-Pytorch)
This directory has two directories:

[1] [LoadData](3. AlexNet-Pytorch/1. LoadData)
This cell tells you how to use load larger data set of Pytorch and have five crops at Pytorch.

> ![](fils/02.png)
>
> 

[2] [AlexNet_Pytorch](3. AlexNet-Pytorch/2. AlexNet_Pytorch)

This cell, Using Pytorch to create the AlexNet model and saved the model.

> ![](fils/03.png)



**Keras**:
The code in this directory [AlexNet-Keras](4. AlexNet-Keras)
This director has tow files.

[1] [Load Data set](4. AlexNet-Keras/1. LoadData.ipynb)

This cell tells you how to use load larger data set of Keas.

> ![](fils/04.png)

[2] [AlexNet-Keras](4. AlexNet-Keras/2. AlexNet-Keras.ipynb)

Using Keras to create the AlexNet model and save the model in h5py.

> ![](../../../picture/209.png)