import tensorflow as tf
from loadData import LoadData, SplitData
import matplotlib.pyplot as plt
import numpy as np

class DeConvolution:
    """
    DeConvolution step:
    =====================================================
    1. forward to target layers.
    2. Unpool, if hidden layer have max pooling.
    3. DeActivate, can using ReLu to make sure value > 0.
    4. Deconvolution or say convolution transpose.
    
    CONV==>RELU==>POOL==>UNPOOL==>RELU==>DECONV
    ======================================================
    """
    def __init__(self, file_dir, model_dir, Load_samples=None, batch_size=9, test_rate=0.3):
        """
        :param file_dir (string): images dir like '../train'
        :param model_dir (string): AlexNet save model path.
        :param Load_samples (int,None): load samples limit, if equal None, means loads all samples.
        :param batch_size (int): data batch size.
        :param test_rate (float): test samples rate.
        """
        self.sess = tf.Session()
        self.file_dir = file_dir
        self.model_dir = model_dir
        self.Load_samples = Load_samples
        self.test_rate = test_rate
        self.batch_size = batch_size

    def load_images(self):
        """
        load visualization samples.
        :return
            next_element_train (generator): can iter images and labels.
        """
        loader = LoadData(self.batch_size, self.sess)
        next_element_train = loader.get_data(self.train_files)
        return next_element_train

    def LoadWandb(self):
        """
        Reloaded weights and bias in save model_dir.
        """
        new_saver = tf.train.import_meta_graph(self.model_dir + '.meta')
        new_saver.restore(self.sess, self.model_dir)
        graph = tf.get_default_graph()
        self.W1 = graph.get_tensor_by_name('W%d:0' % (1))
        self.b1 = graph.get_tensor_by_name('b%d:0' % (1))
        self.W2 = graph.get_tensor_by_name('W%d:0' % (2))
        self.b2 = graph.get_tensor_by_name('b%d:0' % (2))
        self.W3 = graph.get_tensor_by_name('W%d:0' % (3))
        self.b3 = graph.get_tensor_by_name('b%d:0' % (3))
        self.W4 = graph.get_tensor_by_name('W%d:0' % (4))
        self.b4 = graph.get_tensor_by_name('b%d:0' % (4))
        self.W5 = graph.get_tensor_by_name('W%d:0' % (5))
        self.b5 = graph.get_tensor_by_name('b%d:0' % (5))

    def _Conv(self, data, W, b, s, p, name):
        """
        Convolution layers.

        :param data (tensor): convolution data.
        :param W (tensor): convolution kernel.
        :param b (tensor): convolution bias.
        :param s (tuple,list): convolution strides.
        :param p (string): convolution mode, can choose "SAME","VALID".
        :param name (string): Tensor graph name.
        :return C (tensor): Convolution result.
        """
        C = tf.nn.conv2d(input=data, filter=W, strides=s, padding=p, name=name) + b
        return C

    def _Pool(self, value, k, s, p, name):
        """
        Pooling layers.
        Note:
        =========================================================================
        Implementation Unpool can using following function
        tf.nn.max_pool_with_argmax:
            return max value and max value index.
            ==>
            https://www.tensorflow.org/api_docs/python/tf/nn/max_pool_with_argmax
        ==========================================================================

        :param value (tensor): pooling value.
        :param k (tensor): pooling kernel.
        :param s (tuple,list): pooling strides.
        :param p (string): pooling mode, can choose "SAME","VALID".
        :param name (string): Tensor graph name.
        """
        pooled, ind = tf.nn.max_pool_with_argmax(input=value, ksize=k, strides=s, padding=p, name=name)
        return pooled, ind

    def _Unpool_with_with_argmax(self, pooled, ind, out_size):
        """
        Unpool layer.
        Note:
        ======================================================================================
        In Tensorflow, the tensor can view but can't change,or say max_index-max_value change.
        So, we need it's function:tf.compat.v1.scatter_nd_update
        ==>
        https://www.tensorflow.org/api_docs/python/tf/scatter_nd_update
        Update max value to max index
        ======================================================================================

        :param pooled (tensor): pooling layer result.
        :param ind (tensor): pooling layer max value index.
        :param out_size (tensor): unpooling result.
        :return
            un_pool (tensor): unpool result.
        """
        m, h, w, c = out_size
        _, h_, w_, c_ = pooled.shape.as_list()

        ref = tf.Variable(tf.zeros([m * h * w * c]))
        pooled_ = tf.reshape(pooled, [m * h_ * w_ * c_])
        ind_ = tf.reshape(ind, [m * h_ * w_ * c_])
        self.sess.run(ref.initializer)

        # expand dims.
        indices = tf.expand_dims(ind_, axis=1)
        # update tensor.
        un_pool = tf.compat.v1.scatter_nd_update(ref, indices=indices, updates=pooled_)
        # reshape to (m, h, w, c)
        un_pool = tf.reshape(un_pool, (m, h, w, c))

        return un_pool


    def forward(self, data, layers='layer1'):
        """
        forward propagation.

        :param data (tensor): training data set,shape [batch_size,227,227,3].
        :param layers (string): control forward step.

        """
        C1 = self._Conv(data, self.W1, self.b1, [1, 4, 4, 1], "VALID", 'Conv_1')
        R1 = tf.nn.relu(C1)
        pooled_1, max_index_1 = self._Pool(R1, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', 'Pool_1')
        if layers == 'layer1':
            return pooled_1, max_index_1
        C2 = self._Conv(pooled_1, self.W2, self.b2, [1, 1, 1, 1], 'SAME', 'Conv_2')
        R2 = tf.nn.relu(C2)
        pooled_2, max_index_2 = self._Pool(R2, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', 'Pool_2')
        if layers == 'layer2':
            return pooled_2, max_index_2, max_index_1
        C3 = self._Conv(pooled_2, self.W3, self.b3, [1, 1, 1, 1], 'SAME', 'Conv_3')
        R3 = tf.nn.relu(C3)
        if layers == 'layer3':
            return R3, max_index_2, max_index_1
        C4 = self._Conv(R3, self.W4, self.b4, [1, 1, 1, 1], 'SAME', 'Conv_4')
        R4 = tf.nn.relu(C4)
        if layers == 'layer4':
            return R4, max_index_2, max_index_1
        C5 = self._Conv(R4, self.W5, self.b5, [1, 1, 1, 1], 'SAME', 'Conv_5')
        R5 = tf.nn.relu(C5)
        pooled_5, max_index_5 = self._Pool(R5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', 'Pool_5')
        if layers == 'layer5':
            return pooled_5, max_index_5, max_index_2, max_index_1

    def _Deconv_hanlder(self, pooled, max_index, un_pool_out, W, out_shape, strides, padding):
        """
        :param pooled (tensor): max_pool layer result
        :param max_index (tensor): max_pool layer max index.
        :param un_pool_out (list or tuple): unpool layer out shape.
        :param W (tensor): weights.
        :param out_shape (list or tuple): convolution transpose out shape.
        :param strides (list or tuple):  convolution transpose strides.
        :param padding (string): convolution transpose padding,can choose "VALID" or "SAME"
        :return
            C_transpose_ (tensor): deConvolution result.
        """
        un_pool_ = self._Unpool_with_with_argmax(pooled, max_index, un_pool_out)
        deReLu_ = tf.nn.relu(un_pool_)
        C_transpose_ = tf.nn.conv2d_transpose(deReLu_, W, out_shape, strides, padding)
        return C_transpose_

    def deconv1(self, pooled_1, max_index_1, m):
        C_transpose_1 = self._Deconv_hanlder(pooled_1, max_index_1, [m, 55, 55, 96],
                                             self.W1, [m, 227, 227, 3], [1, 4, 4, 1], "VALID")
        return C_transpose_1

    def deconv2(self, pooled_2, max_index_2, max_index_1, m):
        C_transpose_2 = self._Deconv_hanlder(pooled_2, max_index_2, [m, 27, 27, 256],
                                             self.W2, [m, 27, 27, 96], [1, 1, 1, 1], 'SAME')
        normal = self.deconv1(C_transpose_2, max_index_1, m)
        return normal

    def deconv3(self, R3, max_index_2, max_index_1, m):
        C_transpose_3 = tf.nn.conv2d_transpose(R3, self.W3, [m, 13, 13, 256], [1, 1, 1, 1], "SAME")
        normal = self.deconv2(C_transpose_3, max_index_2, max_index_1, m)
        return normal

    def deconv4(self, R4, max_index_2, max_index_1, m):
        C_transpose_4 = tf.nn.conv2d_transpose(R4, self.W4, [m, 13, 13, 384], [1, 1, 1, 1], "SAME")
        normal = self.deconv3(C_transpose_4, max_index_2, max_index_1, m)
        return normal

    def deconv5(self, pooled_5, max_index_5, max_index_2, max_index_1, m):
        C_transpose_5 = self._Deconv_hanlder(pooled_5, max_index_5, [m, 13, 13, 256],
                                             self.W5, [m, 13, 13, 384], [1, 1, 1, 1], 'SAME')
        normal = self.deconv4(C_transpose_5, max_index_2, max_index_1, m)
        return normal

    def _Deconv(self, layer='layer1'):
        """
        deconv...
        :param layer (string): control deConvolution layers.[layer1-layer5]
        :return
            normal (tensor): deconv result. shape [batch_size,227,227,3]
            images (ndarray): original images. shape [batch_size,227,227,3]
        """
        next_element_train = self.load_images()
        images, _ = self.sess.run(next_element_train)

        m = images.shape[0]
        if layer == 'layer1':
            pooled_1, max_index_1 = self.forward(images, layers=layer)
            normal = self.deconv1(pooled_1, max_index_1, m)
            return normal, images

        if layer == 'layer2':
            pooled_2, max_index_2, max_index_1 = self.forward(images, layers=layer)
            normal = self.deconv2(pooled_2, max_index_2, max_index_1, m)
            return normal, images
        if layer == 'layer3':
            R3, max_index_2, max_index_1 = self.forward(images, layer)
            normal = self.deconv3(R3, max_index_2, max_index_1, m)
            return normal, images
        if layer == 'layer4':
            R4, max_index_2, max_index_1 = self.forward(images, layer)
            normal = self.deconv4(R4, max_index_2, max_index_1, m)
            return normal, images

        if layer == 'layer5':
            pooled_5, max_index_5, max_index_2, max_index_1 = self.forward(images, layer)
            normal = self.deconv5(pooled_5, max_index_5, max_index_2, max_index_1, m)
            return normal, images


    def PLot(self, layer, images,name):
        """
        Plot de-images.
        :param images (ndarray): de-images or original images.
        :param name (string): save file name.

        """
        figure = plt.figure(figsize=(20,20))
        row,col = 3,3
        if row * col != self.batch_size:
            raise KeyError('plot row * col != batch_size')
        for i in range(row*col):
            ax = figure.add_subplot(row,col,i+1)
            img = images[i]
            ax.imshow(img)
            ax.set_title(layer)
            ax.set_xticks(())
            ax.set_yticks(())
        plt.savefig(name+'_'+layer+'.jpg')



    def _Normal_Range(self, filter_in):
        """
        Normal ones.
        ===========================================
        New_value = (old_Value - min) / (max - min)
        ===========================================

        :param filter_in (ndarray): [h_size,w_size,channels]
        :return: normal ones filter_in.
        """
        f_min = np.amin(filter_in)
        f_max = np.amax(filter_in)

        return (filter_in - f_min) * 1.0 / (f_max - f_min + 1e-5) * 255.0

    def _Normal_Std(self,filter_in):
        """
        Normalization of conv2d filters for visualization:
        ===========================================================================
        https://github.com/jacobgil/keras-filter-visualization/blob/master/utils.py
        ===========================================================================

        :param filter_in (ndarray): [h_size,w_size,channels]
        :return:
        """
        x = filter_in
        x -= x.mean()
        x /= (x.std() + 1e-5)
        # make most of the value between [-0.5, 0.5]
        x *= 0.1
        # move to [0, 1]
        x += 0.5
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')

        return x


    def start_deConv(self, de_layers,norml_mode='range'):
        """
        Start function to running deConv...
        :param de_layers (list or tuple): include layers. [layer1-layer5]
        :param norml_mode (string):
            normalization of deConv-images,can choose "range:normal-ones", "std:normal-stand"
        """
        self.LoadWandb()
        split_data = SplitData(self.file_dir, Load_samples=self.Load_samples, test_rate=self.test_rate)
        self.train_files, _ = split_data()

        for layer in de_layers:
            normal, original = self._Deconv(layer)
            normal_ = self.sess.run(normal)

            # mapping in Normal func.
            if norml_mode == 'range':
                normal_ = map(self._Normal_Range,normal_)
            elif norml_mode == 'std':
                normal_ = map(self._Normal_Std, normal_)
            normal_ = np.array(list(normal_))

            # plot
            self.PLot(layer, normal_, 'DeConv')
            self.PLot(layer,original,'Original')
            print('%s is over..'%layer)


if __name__ == "__main__":
    model_dir = 'model/alexNet'
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    de_Conv = DeConvolution(file_dir, model_dir)
    de_Conv.start_deConv(['layer1', 'layer2', 'layer3', 'layer4', 'layer5'],norml_mode='std')
