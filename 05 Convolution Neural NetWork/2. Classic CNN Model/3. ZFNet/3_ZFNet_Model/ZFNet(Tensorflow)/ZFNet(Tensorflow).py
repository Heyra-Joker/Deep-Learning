import tensorflow as tf
from loadData_TF import SplitData, LoadData


class ZFNet:
    """
    Implementation AlexNet.
    AlexNet paper:
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    Note:
          in this code, not using data augmented.in the original paper, need to down sample
          images to 256 and crop to (224,224), to flip.
    """

    def __init__(self, file_dir, Load_samples=100, test_rate=0.3, n_classes=2, batch_size=50):
        """
        :param file_dir (string): training data dir.
        :param Load_samples (int or None): loaded Number of total data,if it's None,load all files.
        :param test_rate (float): split test data rate.default 0.3
        :param n_classes (int):
            classes of labels,notice,if it's not equal 1, then you need modify Cost function.
            original Cost function is sigmoid_cross_entropy_with_logits.
        :param batch_size (int): data batch size.
        """
        self.n_classes = n_classes
        self.file_dir = file_dir
        self.Load_samples = Load_samples
        self.test_rate = test_rate
        self.batch_size = batch_size

    def init_parameters(self):
        """
        Initialization parameters.

        :return params (tuple): include weights and bias,notice,weights have fully connect layer.
        """
        init_W = tf.initializers.glorot_normal()
        init_b = tf.initializers.zeros()
        # Convolution layers...
        W1 = tf.get_variable('W1', [7, 7, 3, 96], initializer=init_W)
        b1 = tf.get_variable('b1', [1, 1, 96], initializer=init_b)
        W2 = tf.get_variable('W2', [5, 5, 96, 256], initializer=init_W)
        b2 = tf.get_variable('b2', [1, 1, 256], initializer=init_b)
        W3 = tf.get_variable('W3', [3, 3, 256, 384], initializer=init_W)
        b3 = tf.get_variable('b3', [1, 1, 384], initializer=init_b)
        W4 = tf.get_variable('W4', [3, 3, 384, 384], initializer=init_W)
        b4 = tf.get_variable('b4', [1, 1, 384], initializer=init_b)
        W5 = tf.get_variable('W5', [3, 3, 384, 256], initializer=init_W)
        b5 = tf.get_variable('b5', [1, 1, 256], initializer=init_b)

        # Fully connect layers....
        W6 = tf.get_variable('W6', [6 * 6 * 256, 4096], initializer=init_W)
        b6 = tf.get_variable('b6', [1, 4096], initializer=init_b)
        W7 = tf.get_variable('W7', [4096, 4096], initializer=init_W)
        b7 = tf.get_variable('b7', [1, 4096], initializer=init_b)
        W8 = tf.get_variable('W8', [4096, self.n_classes], initializer=init_W)
        b8 = tf.get_variable('b8', [1, self.n_classes], initializer=init_b)
        params = (W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8)

        return params

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

        :param value (tensor): pooling value.
        :param k (tensor): pooling kernel.
        :param s (tuple,list): pooling strides.
        :param p (string): pooling mode, can choose "SAME","VALID".
        :param name (string): Tensor graph name.
        :return P (tensor): Pooling result.
        """
        P = tf.nn.max_pool(value=value, ksize=k, strides=s, padding=p, name=name)
        return P

    def _Lrn(self, input_, name):
        """
        Local response normal.
        :param input_ (tensor): LRN input.
        :param name (string): tensor graph name.
        :return L (tensor): LRN result.
        """
        L = tf.nn.local_response_normalization(input=input_, name=name)
        return L

    def _Fc(self, value, W, b):
        """
        Fully connect layer.
        :param value (tensor): fc input value.
        :param W (tensor): fc weights.
        :param b (tensor): fc bias.
        :return F(tensor): fc result.
        """
        F = tf.matmul(value, W) + b
        return F

    def _Drop(self, value, rate):
        """
        Dropout layer.

        :param value (tensor): dropout value.
        :param rate (float): dropout rate,it's equal 1-keep_prob.
        :return D (tensor): dropout result.
        """
        D = tf.nn.dropout(value, rate=rate)
        return D

    def forward(self, data, params, rate):
        """
        forward propagation.
        :param data (tensor): training data set.
        :param params (tuple,tensor): weights and bias.
        :param rate (float): dropout rate.
        :return out (tensor): forward result, the shape is (batch,n_classes).
        """
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8 = params
        # CONV1
        C1 = self._Conv(data, W1, b1, [1, 2, 2, 1], "VALID", 'Conv_1')
        R1 = tf.nn.relu(C1)
        P1 = self._Pool(R1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', 'Pool_1')
        L1 = self._Lrn(P1, 'LRN_1')
        # CONV2
        C2 = self._Conv(L1, W2, b2, [1, 2, 2, 1], 'VALID', 'Conv_2')
        R2 = tf.nn.relu(C2)
        P2 = self._Pool(R2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', 'Pool_2')
        L2 = self._Lrn(P2, 'LRN_2')
        # CONV3
        C3 = self._Conv(L2, W3, b3, [1, 1, 1, 1], 'SAME', 'Conv_3')
        R3 = tf.nn.relu(C3)
        # CONV4
        C4 = self._Conv(R3, W4, b4, [1, 1, 1, 1], 'SAME', 'Conv_4')
        R4 = tf.nn.relu(C4)
        # CONV5
        C5 = self._Conv(R4, W5, b5, [1, 1, 1, 1], 'SAME', 'Conv_5')
        R5 = tf.nn.relu(C5)
        P5 = self._Pool(R5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', 'Pool_5')
        print(P5)
        # flatten
        flatten = tf.reshape(P5, [-1, 6 * 6 * 256])
        # FC6
        F6 = self._Fc(flatten, W6, b6)
        R6 = tf.nn.relu(F6)
        D6 = self._Drop(R6, rate=rate)
        # FC7
        F7 = self._Fc(D6, W7, b7)
        R7 = tf.nn.relu(F7)
        D7 = self._Drop(R7, rate=rate)
        # Out
        out = self._Fc(D7, W8, b8)

        return out

    def fit(self, lr, epochs, drop_rate):
        """
        fitting model.
        :param lr (float): learning rate.
        :param epochs (int): Iterate of epoch.
        :param drop_rate (float): dropout rate. rate = 1 - keep_prob.
        :return:
        """

        # create placeholder
        data = tf.placeholder(tf.float32, [None, 225, 225, 3], name='Input')
        labels = tf.placeholder(tf.float32, [None, 1], name='Labels')
        rate = tf.placeholder(tf.float32, name='rate')

        # build model.
        params = self.init_parameters()
        out = self.forward(data, params, rate)
        Cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=labels))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(Cost)
        # score.
        predict = tf.round(tf.sigmoid(out))
        equal = tf.equal(labels, predict)
        correct = tf.cast(equal, tf.float32)
        accuracy = tf.reduce_mean(correct)

        # split date..
        split_data = SplitData(self.file_dir, Load_samples=self.Load_samples, test_rate=self.test_rate)
        train_files, test_files = split_data()

        self.N_train = len(train_files) // self.batch_size
        self.N_test = len(test_files) // self.batch_size

        # Saver..
        saver = tf.train.Saver()
        tf.add_to_collection('pre_network', out)
        init = tf.global_variables_initializer()

        # training...
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(epochs):
                loader = LoadData(self.batch_size, sess)
                next_element_train = loader.get_data(train_files)
                # running all training set...
                count = 1
                while 1:
                    try:
                        images, target = sess.run(next_element_train)
                        print('Training {}/{} \r'.format(count, self.N_train), end='', flush=True)
                        count += 1
                    except tf.errors.OutOfRangeError:
                        break
                    else:
                        _ = sess.run(optimizer, feed_dict={data: images, labels: target, rate: drop_rate})

                acc_train, loss_train = self.Caculate(sess, train_files, accuracy, Cost, data, labels, rate, 'train')
                acc_test, loss_test = self.Caculate(sess, test_files, accuracy, Cost, data, labels, rate, 'test')
                print('[{}/{}] train loss:{:.4f} - train acc:{:.4f} - test loss:{:.4f} - test acc:{:.4f}'.format(
                    epoch + 1, epochs, loss_train, acc_train, loss_test, acc_test
                ))
                if acc_train >= 0.980:
                    break
            # Saver ....
            saver.save(sess, 'model/ZFNet')

    def Caculate(self, sess, files, accuracy_tensor, Cost_tensor, data_placeholder,
                 labels_placeholder, rate_placeholder, model='train'):
        """
        Calculate accuracy and loss.
        :param sess : tensor graph.
        :param files (ndarray): score data.
        :param accuracy_tensor (tensor): accuracy tensor function graph.
        :param Cost_tensor (tensor): cost tensor function graph.
        :param data_placeholder (tensor): data placeholder.
        :param labels_placeholder (tensor): labels placeholder.
        :param rate_placeholder (tensor): rate placeholder.
        :param model (string): can choose 'train' or 'test' to scored model.
        :return acc_mean,loss_mean (float): mean accuracy and mean loss.
        """
        if model == 'train':
            N = self.N_train
        else:
            N = self.N_test
        loader = LoadData(self.batch_size, sess)
        next_element_ = loader.get_data(files)
        acc, loss, count = 0, 0, 1

        while 1:
            try:
                images, labels = sess.run(next_element_)
                print('Score {}/{} \r'.format(count, N), end='', flush=True)
                count += 1
            except tf.errors.OutOfRangeError:
                break
            else:
                acc_, loss_ = sess.run([accuracy_tensor, Cost_tensor], feed_dict={data_placeholder: images,
                                                                                  labels_placeholder: labels,
                                                                                  rate_placeholder: 0})
                acc += acc_
                loss += loss_

        acc_mean = acc / count
        loss_mean = loss / count

        return acc_mean, loss_mean


if __name__ == '__main__':
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    zfnet = ZFNet(file_dir, Load_samples=100, test_rate=0.2, n_classes=1, batch_size=20)
    zfnet.fit(lr=1e-3, epochs=2, drop_rate=0.)

