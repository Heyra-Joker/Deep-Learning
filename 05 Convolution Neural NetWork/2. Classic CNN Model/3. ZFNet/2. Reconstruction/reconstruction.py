# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from AlexDeConv import *

def OpenImage(file_dir,new_width,new_height,top_num=9):

    files = os.listdir(file_dir)
    np.random.shuffle(files)
    files = files[:top_num]
    imgs = np.zeros((top_num,new_height,new_width,3))
    for index,file in enumerate(files):
        img_file = os.path.join(file_dir,file)
        img = Image.open(img_file)
        img = img.resize((new_width,new_height))
        img = np.array(img)
        img = img.reshape((new_height,new_width,3))
        imgs[index,...] += img

    imgs = tf.convert_to_tensor(imgs,tf.float32)
    return imgs

def Plot_img(index,imgs,original_imgs,row=3,col=3):
    fig = plt.figure(figsize=(5,5))
    plt.subplots_adjust(left=0., bottom=0., right=1., top=1.,hspace=0,wspace=0)
    for i in range(1,row*col+1):
        ax = plt.subplot(row,col,i)
        plt.imshow(imgs[i-1])
        plt.xticks(())
        plt.yticks(())

    plt.savefig('deconv{}.jpg'.format(index))
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(5,5))
    plt.subplots_adjust(left=0., bottom=0., right=1., top=1.,hspace=0,wspace=0)
    for i in range(1,row*col+1):
        ax = plt.subplot(row,col,i)
        plt.imshow(original_imgs[i-1])
        plt.xticks(())
        plt.yticks(())

    plt.savefig('original{}.jpg'.format(index))
    plt.show()
    plt.close()


if __name__ == "__main__":

    visualization_path = '/Users/huwang/Joker/Data_Set/catVSdot/test'
    model_dir = 'Model_AlexNet/alexnet'
    sess = tf.Session()
    for i in range(1,6):
        imgs = OpenImage(visualization_path,227,227)
        loaderModel = Deconv(sess,model_dir,imgs)
        if i == 1:
            res = loaderModel.deconv1()
        elif i == 2:
            res = loaderModel.deconv2()
        elif i == 3:
            res = loaderModel.deconv3()
        elif i == 4:
            res = loaderModel.deconv4()
        elif i == 5:
            res = loaderModel.deconv5()
        res2 = sess.run(imgs).astype('uint8')
        Plot_img(i,res,res2,3,3)
    sess.close()
    