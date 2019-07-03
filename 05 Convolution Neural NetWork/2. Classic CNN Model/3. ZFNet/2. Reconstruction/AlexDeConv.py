import tensorflow as tf
import numpy as np

from AlexConv import Load_Model


class Deconv(Load_Model):
    def __init__(self, sess, model_dir, imgs):
        super().__init__(sess, model_dir, imgs)
        
    def UnPool(self,value,index,output_shape):
        p_b,p_h,p_w,p_c = value.shape
        _,h,w,c = output_shape
        zeros = np.zeros((p_b*h*w*c))
        value_ = value.reshape((p_b,p_h*p_w*p_c))
        index_ = index.reshape((p_b,p_h*p_w*p_c))
        for i in range(p_b):
            zeros[index_[i]] += value_[i]
        zeros = zeros.reshape((p_b,h,w,c))
        zeros = tf.convert_to_tensor(zeros,dtype=tf.float32)
        return zeros

    
    def Normal(self,deconv):
        deconv_out = self.sess.run(deconv)
        deconv_out = ((deconv_out - deconv_out.min())*255) / (deconv_out.max() - deconv_out.min())
        return deconv_out.astype('uint8')

    def deconv1(self,is_cal_pool_value=True,value=None):
        if is_cal_pool_value:
            _ = self.conv1()
            value,index = self.sess.run(self.argmaxpool_1)
        else:
            _ = self.conv1()
            _,index = self.sess.run(self.argmaxpool_1)
            value = self.sess.run(value)

        unpool_ = self.UnPool(value,index,self.layers_param['conv1_outshape'])
        output_shape = self.layers_param['input_shape']
        strides = self.layers_param['conv1_strides']
        deconv_1 = tf.nn.conv2d_transpose(unpool_,self.W1,output_shape,strides=strides,padding="VALID")
        Normal_1 = self.Normal(deconv_1)

        return Normal_1
    def deconv2(self,is_cal_pool_value=True,value=None):
        if is_cal_pool_value:
            _ = self.conv2()
            value,index = self.sess.run(self.argmaxpool_2)
        else:
            _ = self.conv2()
            _,index = self.sess.run(self.argmaxpool_2)
            value = self.sess.run(value)

        unpool_ = self.UnPool(value,index,self.layers_param['conv2_outshape'])
        output_shape = self.layers_param['pool1_outshape']
        strides = self.layers_param['conv2_strides']
        deconv_2 = tf.nn.conv2d_transpose(unpool_,self.W2,output_shape,strides=strides,padding="SAME")
        Normal_1 = self.deconv1(is_cal_pool_value=False,value=deconv_2)
        return Normal_1

    def deconv3(self,is_cal_pool_value=True,value=None):

        if is_cal_pool_value:
            value = self.sess.run(self.conv3())

        output_shape = self.layers_param['pool2_outshape']
        strides = self.layers_param['conv3_strides']
        deconv_3 = tf.nn.conv2d_transpose(value,self.W3,output_shape,strides=strides,padding="SAME")
        Normal_1 = self.deconv2(False,value=deconv_3)
        return Normal_1
    
    def deconv4(self,is_cal_pool_value=True,value=None):
        if is_cal_pool_value:
            value = self.sess.run(self.conv4())

        output_shape = self.layers_param['conv3_outshape']
        strides = self.layers_param['conv4_strides']
        deconv_4 = tf.nn.conv2d_transpose(value,self.W4,output_shape,strides=strides,padding="SAME")
        Normal_1 = self.deconv3(False,value=deconv_4)
        return Normal_1

    def deconv5(self):
        _ = self.conv5()
        value,index = self.sess.run(self.argmaxpool_5)
        unpool_ = self.UnPool(value,index,self.layers_param['conv5_outshape'])
        output_shape = self.layers_param['conv4_outshape']
        strides = self.layers_param['conv5_strides']
        deconv_5 = tf.nn.conv2d_transpose(unpool_,self.W5,output_shape,strides=strides,padding="SAME")
        Normal_1 = self.deconv4(False,value=deconv_5)
        return Normal_1



        
        


            





        

    

    



