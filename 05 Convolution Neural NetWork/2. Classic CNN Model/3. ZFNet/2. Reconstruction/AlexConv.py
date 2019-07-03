import tensorflow as tf
import numpy as np


class Load_Model:
    def __init__(self,sess,model_dir,imgs):
        self.sess = sess
        self.model_dir = model_dir
        self.imgs = imgs
        m = self.sess.run(self.imgs).shape[0]
        self.layers_param = {'input_shape':(m,227,227,3),
                            'conv1_strides':(1,4,4,1),
                            'conv1_outshape':(m,55,55,96),
                            'pool1_ksize':(1,3,3,1),
                            'pool1_strides':(1,2,2,1),
                            'pool1_outshape':(m,27,27,96),
                            'conv2_strides':(1,1,1,1),
                            'conv2_outshape':(m,27,27,256),
                            'pool2_ksize':(1,3,3,1),
                            'pool2_strides':(1,2,2,1),
                            'pool2_outshape':(m,13,13,256),
                            'conv3_strides':(1,1,1,1),
                            'conv3_outshape':(m,13,13,384),
                            'conv4_strides':(1,1,1,1),
                            'conv4_outshape':(m,13,13,384),
                            'conv5_strides':(1,1,1,1),
                            'conv5_outshape':(m,13,13,256),
                            'pool5_ksize':(1,3,3,1),
                            'pool5_strides':(1,2,2,1),
                            'pool5_outshape':(m,6,6,256) 
                            }
        self.Loader()

    def Loader(self):
        self.param = {}
        new_saver = tf.train.import_meta_graph(self.model_dir + '.meta')
        new_saver.restore(self.sess, self.model_dir)
        graph = tf.get_default_graph()
        
        for i in range(5):
            W = graph.get_tensor_by_name('W%d:0'%(i+1))
            b = graph.get_tensor_by_name('b%d:0'%(i+1))
            self.param['W%d'%(i+1)] = W
            self.param['b%d'%(i+1)] = b

    def _conv2d(self,input_,kernel,strides=[1,1,1,1],padding="VALID"):
        conv2d_ = tf.nn.conv2d(input_,kernel,strides,padding)
        return conv2d_
    
    def _argmaxpool(self,value,ksize,strides=[1,1,1,1],padding="VALID"):
        argmaxpool_ = tf.nn.max_pool_with_argmax(value,ksize,strides,padding)
        return argmaxpool_
    
    def _lrn(self,input_,depth_radius=5,bias=2,alpha=1e-4,beta=0.75):
        lrn_ = tf.nn.lrn(input_,depth_radius,bias,alpha,beta)
        return lrn_

    def conv1(self):
        self.W1 = self.param['W1']
        self.b1 = self.param['b1']
        conv2d_ = self._conv2d(input_= self.imgs,kernel=self.W1,strides=self.layers_param['conv1_strides']) + self.b1
        relu_ = tf.nn.relu(conv2d_)
        self.argmaxpool_1 = self._argmaxpool(value=relu_,ksize=self.layers_param['pool1_ksize'],strides=self.layers_param['pool1_strides'])
        lrn_ = self._lrn(self.argmaxpool_1[0])
        return lrn_

    def conv2(self):
        self.W2 = self.param['W2']
        self.b2 = self.param['b2']
        lrn_1 = self.conv1()
        conv2d_ = self._conv2d(input_=lrn_1,kernel=self.W2,strides=self.layers_param['conv2_strides'],padding="SAME") + self.b2
        relu_ = tf.nn.relu(conv2d_)
        self.argmaxpool_2 = self._argmaxpool(value=relu_,ksize=self.layers_param['pool2_ksize'],
                                       strides=self.layers_param['pool2_strides'])
        lrn_ = self._lrn(self.argmaxpool_2[0])
        return lrn_

    def conv3(self):
        self.W3 = self.param['W3']
        self.b3 = self.param['b3']
        lrn_2 = self.conv2()
        conv2d_3 = self._conv2d(input_=lrn_2,kernel=self.W3,strides=self.layers_param['conv3_strides'],padding="SAME") + self.b3
        relu_ = tf.nn.relu(conv2d_3)
        return relu_

    def conv4(self):
        self.W4 = self.param['W4']
        self.b4 = self.param['b4']
        relu_3 = self.conv3()
        conv2d_4 = self._conv2d(input_=relu_3,kernel=self.W4,strides=self.layers_param['conv4_strides'],padding="SAME") + self.b4
        relu_ = tf.nn.relu(conv2d_4)
        return relu_

    def conv5(self):
        self.W5 = self.param['W5']
        self.b5 = self.param['b5']
        relu_4 = self.conv4()
        conv2d_ = self._conv2d(input_=relu_4,kernel=self.W5,strides=self.layers_param['conv5_strides'],padding="SAME") + self.b5
        relu_ = tf.nn.relu(conv2d_)
        self.argmaxpool_5 = self._argmaxpool(value=relu_,ksize=self.layers_param['pool5_ksize'],
                                       strides=self.layers_param['pool5_strides'])
        return self.argmaxpool_5



if __name__ == "__main__":
    pass

    

    
