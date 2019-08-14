import os
import numpy as np
import tensorflow as tf

from LoadData import LoadFiles, LoadData, resize_image

"""
##########################
# Regression of OverFeat # 
##########################

The Paper addr:
-----------------------------------
https://arxiv.org/pdf/1312.6229.pdf
-----------------------------------

Also, you can see chinese paper addr:
-----------------------------------------
http://www.chenzhaobin.com/notes/overfeat
-----------------------------------------

Note:
----
    In this code, I do not used "offset pooling" and "multi-scal".
    Please see OverFeat(Jupyter) for details.
"""


class OverFeat:
    def __init__(self, sess, model_dir):
        """
        Arguments:
        ---------
            sess: tensorflow graph.
            model_dir(str): trained model path.
        """
        self.sess = sess
        self.model_dir = model_dir

    def load_w_b(self):
        """
        load target parameters.
        ----------------------------------------------
        https://tensorflow.google.cn/guide/saved_model
        ----------------------------------------------
        """
        self.W1 = tf.get_variable("W1", (11, 11, 3, 96))
        self.b1 = tf.get_variable("b1", (1, 1, 96))
        self.W2 = tf.get_variable("W2", (5, 5, 96, 256))
        self.b2 = tf.get_variable("b2", (1, 1, 256))
        self.W3 = tf.get_variable("W3", (3, 3, 256, 512))
        self.b3 = tf.get_variable("b3", (1, 1, 512))
        self.W4 = tf.get_variable("W4", (3, 3, 512, 1024))
        self.b4 = tf.get_variable("b4", (1, 1, 1024))
        self.W5 = tf.get_variable("W5", (3, 3, 1024, 1024))
        self.b5 = tf.get_variable("b5", (1, 1, 1024))

        saver = tf.train.Saver([
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4,
            self.b4, self.W5, self.b5
        ])
        # restore target weights and bias.
        saver.restore(self.sess, self.model_dir)

    def init_w_b(self):
        """
        initlization layer 6 - layer 8 weights and bias.
        """
        init_Weights = tf.initializers.glorot_normal()
        init_bias = tf.initializers.zeros()
        self.W6 = tf.get_variable('W6', (6, 6, 1024, 4096),
                                  initializer=init_Weights)
        self.b6 = tf.get_variable('b6', (1, 1, 4096), initializer=init_bias)
        self.W7 = tf.get_variable('W7', (1, 1, 4096, 1024),
                                  initializer=init_Weights)
        self.b7 = tf.get_variable('b7', (1, 1, 1024), initializer=init_bias)
        self.W8 = tf.get_variable('W8', (1, 1, 1024, 4),
                                  initializer=init_Weights)
        self.b8 = tf.get_variable('b8', (1, 1, 4), initializer=init_bias)

    def Iou(self, predict_box, true_box):
        """
        Calculate Iou.
        You can see this page:
        ---------------------------------------------------------
        https://blog.csdn.net/u014061630/article/details/82818112
        ---------------------------------------------------------

        Arguments:
        ---------
            predict_box(array): predict bbox.
            true_box(array): true bbox.
        
        Return:
        ------
            iou(array): iou value, the shape like [bs,4].
        """
        in_w = np.minimum(predict_box[:, 2], true_box[:, 2]) - np.maximum(
            predict_box[:, 0], true_box[:, 0])
        in_h = np.minimum(predict_box[:, 3], true_box[:, 3]) - np.maximum(
            predict_box[:, 1], true_box[:, 1])

        # in_w 和in_h 都大于0 则使用面积,否则为0
        inter = np.logical_and(np.greater(in_w, 0), np.greater(
            in_h, 0)) * np.abs(in_w * in_h)

        union = np.multiply((predict_box[:,3] - predict_box[:,1]),(predict_box[:,2] - predict_box[:,0])) + \
            np.multiply((true_box[:,3] - true_box[:,1]),(true_box[:,2] - true_box[:,0])) - inter

        iou = np.divide(inter, union)
        return iou

    def forward(self, data, rate_):
        """
        Build Rregression of OverFeat Net.

        Arguments:
        ----------
            data(tensor): input layer data set. shape [bs,231,231,3]
            rate_(tensor): dropout rate. 
        Return:
        ------
            out(tensor): out layer value. the shape [bs,4]
        """
        # Conv1
        C1 = tf.nn.conv2d(
            data, filter=self.W1, strides=(1, 4, 4, 1),
            padding="VALID") + self.b1
        R1 = tf.nn.relu(C1)
        P1 = tf.nn.max_pool(R1,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv2
        C2 = tf.nn.conv2d(
            P1, filter=self.W2, strides=(1, 1, 1, 1),
            padding="VALID") + self.b2
        R2 = tf.nn.relu(C2)
        P2 = tf.nn.max_pool(R2,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv3
        C3 = tf.nn.conv2d(
            P2, filter=self.W3, strides=(1, 1, 1, 1), padding="SAME") + self.b3
        R3 = tf.nn.relu(C3)

        # Conv4
        C4 = tf.nn.conv2d(
            R3, filter=self.W4, strides=(1, 1, 1, 1), padding="SAME") + self.b4
        R5 = tf.nn.relu(C4)

        # Conv5
        C5 = tf.nn.conv2d(
            R5, filter=self.W5, strides=(1, 1, 1, 1), padding="SAME") + self.b5
        R5 = tf.nn.relu(C5)
        P5 = tf.nn.max_pool(R5,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv6
        C6 = tf.nn.conv2d(
            P5, filter=self.W6, strides=(1, 1, 1, 1),
            padding="VALID") + self.b6
        R6 = tf.nn.relu(C6)
        D6 = tf.nn.dropout(R6, rate=rate_)

        # Conv7
        C7 = tf.nn.conv2d(
            D6, filter=self.W7, strides=(1, 1, 1, 1),
            padding="VALID") + self.b7
        R7 = tf.nn.relu(C7)
        D7 = tf.nn.dropout(R7, rate=rate_)

        # OUT
        C8 = tf.nn.conv2d(
            D7, filter=self.W8, strides=(1, 1, 1, 1),
            padding='VALID') + self.b8
        Out = tf.reshape(C8, shape=(-1, 4))
        return Out

    def Training_Tetsing(self, epoch, epochs, dropout_rate, SAMPLES,
                         iou_thresh, mode):
        """
        Start training or testing.

        Argument:
        --------
            SAMPLES(tuple): incloud current "test/train" data and label.
            iou_thresh(float): iou thresh.
            mode(str): train or test.
        """
        next_element = self.loader_D.get_batchs(SAMPLES[0], SAMPLES[1])
        acc_, loss_, count, n, N = 0, 0, 0, 0, SAMPLES[0].shape[0]
        while True:
            try:
                IMAGES, LABELS = self.sess.run(next_element)
                count += 1
                n += IMAGES.shape[0]
                print('[{}/{}] runing with batch [{}/{}] \r'.format(
                    epoch + 1, epochs, n, N),
                      end='',
                      flush=True)
            except tf.errors.OutOfRangeError:
                if mode == 'test':
                    acc_ /= count
                    loss_ /= count
                    print()
                    return acc_, loss_
                else:
                    break
            else:
                if mode == 'train':
                    # training
                    self.sess.run(self.optimizer,
                                  feed_dict={
                                      self.data_: IMAGES,
                                      self.target_: LABELS,
                                      self.rate_: dropout_rate
                                  })
                else:
                    loss, out_ = self.sess.run(
                        [self.cost, self.out],
                        feed_dict={
                            self.data_: IMAGES,
                            self.target_: LABELS,
                            self.rate_: dropout_rate
                        })

                    loss_ += loss
                    
                    iou_value = self.Iou(out_, LABELS)
                    # if iout greater or equal to thresh,than,accuracy plus 1.
                    iou = np.greater_equal(iou_value, iou_thresh)
                    acc_ += np.mean(iou)

    def Runing(self, lr, epochs, loader_D, iou_thresh, dropout_rate,
               SAMPLES_TRAIN, SAMPLES_TEST, save_model_path, _classes_name):
        """
        Build Rregression OverFeat model.
        -------------------------------
        The part of 6.
        1. create placeholder
        2. handel weights and bias, equivalent change value.
        3. set update params layer 6 - layer8.
        4. build forward.
        5. saver params.
        6. start training or testing.
        """
        self.loader_D = loader_D
        # create placeholder...
        self.data_ = tf.placeholder(
            tf.float32, (None, resize_image[0], resize_image[1], 3), "Input")
        self.target_ = tf.placeholder(tf.int32, (None, 4), "Target_")
        self.rate_ = tf.placeholder(tf.float32, name="DropoutRate")

        # handel weights and bias
        self.load_w_b()
        self.init_w_b()
        # update parameters.
        var_list = [self.W6, self.b6, self.W7, self.b7,self.W8,self.b8]
        # forward propagation
        self.out = self.forward(self.data_, self.rate_)
        self.cost = tf.losses.mean_squared_error(self.out, self.target_)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(
            self.cost, var_list=var_list)

        # Saver..
        saver = tf.train.Saver([self.W1,self.b1,self.W2,self.b2,self.W3,self.b3,
        self.W4,self.b4,self.W5,self.b5,self.W6,self.b6,self.W7,self.b7,self.W8,
        self.b8
        ])

        # init global variables.
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(epochs):
            # training...
            self.Training_Tetsing(epoch,
                                  epochs,
                                  dropout_rate,
                                  SAMPLES_TRAIN,
                                  iou_thresh,
                                  mode='train')
            # verify training
            print('Verify training...')
            acc_train, loss_train = self.Training_Tetsing(epoch,
                                                          epochs,
                                                          dropout_rate,
                                                          SAMPLES_TRAIN,
                                                          iou_thresh,
                                                          mode='test')
            # testing
            acc_test, loss_test = self.Training_Tetsing(epoch,
                                                        epochs,
                                                        0,
                                                        SAMPLES_TEST,
                                                        iou_thresh,
                                                        mode='test')
            print(
                'Cerrent {} {}-{} train-loss: {:.4f} train-acc: {:.4f} test-loss:{:.4f} test-acc:{:.4f}'
                .format(_classes_name, epoch + 1, epochs, loss_train,
                        acc_train, loss_test, acc_test))
            if acc_train >= 0.67:
                # Saver ....
                save_path = saver.save(self.sess, save_model_path)
                print("Model saved in path: %s" % save_path)
                break


def Start(lr,epochs,load_limit=1,test_rate=0.2,batch_size=64,iou_thresh=0.5,dropout_rate=0.5):
    """
    Starting function.
    """
    # load classes
    loader_F = LoadFiles(load_limit=load_limit, test_rate=test_rate)
    # return a generator incloud different classes.
    samples = loader_F.loader()
    for _ in range(load_limit):
        sess = tf.Session()
        # initialization LoadData class.
        loader_D = LoadData(sess, batch_size)
        # yield SAMPLES in currect classes.
        SAMPLE_TRAIN, SAMPLE_TEST, _classes_name = next(samples)
        # Initialization OverFeat of Regression
        over_feat = OverFeat(sess, '../MODELS/OverFeat_C/model.ckpt')
        # Satrt Runing...
        save_model_path = '../MODELS/model_R_{}/OverFeat_R.ckpt'.format(
            _classes_name)
        over_feat.Runing(lr, epochs, loader_D, iou_thresh, dropout_rate,
                         SAMPLE_TRAIN, SAMPLE_TEST, save_model_path,
                         _classes_name)
        # must be closed!
        sess.close()


if __name__ == '__main__':
    Start(lr=0.001,
          epochs=40,
          load_limit=1,
          test_rate=0.2,
          batch_size=32,
          iou_thresh=0.5,
          dropout_rate=0.5)
