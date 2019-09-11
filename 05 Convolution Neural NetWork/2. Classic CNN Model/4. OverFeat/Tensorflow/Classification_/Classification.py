import tensorflow as tf
import numpy as np
from Config import n_classes,resize_image
from LoadData import SplitDataset,LoadData 


"""

##############################
# Classification of OverFeat # 
##############################

The Paper addr:
-----------------------------------
https://arxiv.org/pdf/1312.6229.pdf
-----------------------------------

Also, you can see chinese paper addr:
-----------------------------------------
http://www.chenzhaobin.com/notes/overfeat
-----------------------------------------

OverFeat Net haved 8 layers of simple/faster model and 9 layers of accurate model.

OverFeat Net Impplementation "Classification","Localization","Detection".

The paper author, Used "offset pooling","multi-scale sliding window", "Fully convolution"

Note: 
-----
    1. In the this code, I used simple/faster model.
    2. Tensorflow 1.32.
"""

class OverFeat:
    """
    OverFeat Net
    """
    def __init__(self, batch_size, test_rate):
        """
        Arguments:
        ----------
            batch_size(int): batch size.
            test_rate(float): test samples rate.
        """
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.test_rate = test_rate

    def init_params(self):
        """
        Initialization parameters.

        Return:
        ------
            params: incloud weights and bias.
        """
        # using glorot normal to init value.
        init_weights = tf.initializers.glorot_normal()
        init_bias = tf.initializers.zeros()

        W1 = tf.get_variable("W1", (11, 11, 3, 96), initializer=init_weights)
        b1 = tf.get_variable("b1", (1, 1, 96), initializer=init_bias)
        W2 = tf.get_variable("W2", (5, 5, 96, 256), initializer=init_weights)
        b2 = tf.get_variable("b2", (1, 1, 256), initializer=init_bias)
        W3 = tf.get_variable("W3", (3, 3, 256, 512), initializer=init_weights)
        b3 = tf.get_variable("b3", (1, 1, 512), initializer=init_bias)
        W4 = tf.get_variable("W4", (3, 3, 512, 1024), initializer=init_weights)
        b4 = tf.get_variable("b4", (1, 1, 1024), initializer=init_bias)
        W5 = tf.get_variable("W5", (3, 3, 1024, 1024),
                             initializer=init_weights)
        b5 = tf.get_variable("b5", (1, 1, 1024), initializer=init_bias)
        W6 = tf.get_variable("W6", (6, 6, 1024, 3072),
                             initializer=init_weights)
        b6 = tf.get_variable("b6", (1, 1, 3072), initializer=init_bias)
        W7 = tf.get_variable("W7", (1, 1, 3072, 4096),
                             initializer=init_weights)
        b7 = tf.get_variable("b7", (1, 1, 4096), initializer=init_bias)
        W8 = tf.get_variable("W8", (1, 1, 4096, n_classes),
                             initializer=init_weights)
        b8 = tf.get_variable("b8", (1, 1, n_classes), initializer=init_bias)
        params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8,
                  b8]
        return params

    def forward(self, data, params, rate_):
        """
        Build OverFeat Net.

        Note: it's used simple/faster architecture.
        The Net architecture: Paper Table 1.

        Arguments:
        ---------
            data(tensor): data set.
            params(tensor): weights and bias.
            rate_(float): dropout rate.
        Return:
        ------
            out: out of layers value.
        """
        (W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8,
         b8) = params
        # Conv1
        C1 = tf.nn.conv2d(
            data, filter=W1, strides=(1, 4, 4, 1), padding="VALID") + b1
        R1 = tf.nn.relu(C1)
        P1 = tf.nn.max_pool(R1,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv2
        C2 = tf.nn.conv2d(P1, filter=W2, strides=(1, 1, 1, 1),
                          padding="VALID") + b2
        R2 = tf.nn.relu(C2)
        P2 = tf.nn.max_pool(R2,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv3
        C3 = tf.nn.conv2d(P2, filter=W3, strides=(1, 1, 1, 1),
                          padding="SAME") + b3
        R3 = tf.nn.relu(C3)

        # Conv4
        C4 = tf.nn.conv2d(R3, filter=W4, strides=(1, 1, 1, 1),
                          padding="SAME") + b4
        R5 = tf.nn.relu(C4)

        # Conv5
        C5 = tf.nn.conv2d(R5, filter=W5, strides=(1, 1, 1, 1),
                          padding="SAME") + b5
        R5 = tf.nn.relu(C5)
        P5 = tf.nn.max_pool(R5,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv6
        C6 = tf.nn.conv2d(P5, filter=W6, strides=(1, 1, 1, 1),
                          padding="VALID") + b6
        R6 = tf.nn.relu(C6)
        D6 = tf.nn.dropout(R6, rate=rate_)

        # Conv7
        C7 = tf.nn.conv2d(D6, filter=W7, strides=(1, 1, 1, 1),
                          padding="VALID") + b7
        R7 = tf.nn.relu(C7)
        D7 = tf.nn.dropout(R7, rate=rate_)

        # Conv8
        C8 = tf.nn.conv2d(D7, filter=W8, strides=(1, 1, 1, 1),
                          padding="VALID") + b8
        # reshape to [batch_size, n_classes]
        out = tf.reshape(C8, (-1, n_classes))

        return out

    def top_k(self, out, labels, k):
        """
        Get top-K accuracy return BOOL.
        you can see:
        -------------------------------------------------------------------
        https://tensorflow.google.cn/api_docs/python/tf/math/in_top_k?hl=en
        -------------------------------------------------------------------
        Arguments:
        ----------
            out(tensor): predict result.
            labels(tensor): true labels.
            k(int): top k.

        Return:
        -------
            accuracy: accuracy rate of current K.
        """
        softmax_ = tf.nn.softmax(out, axis=1)
        in_top_k = tf.math.in_top_k(softmax_, labels, k)
        accuracy = tf.reduce_mean(tf.cast(in_top_k, tf.float32))
        return accuracy

    def Go(self,loader,x,y,epoch,epochs,mode,N,cost,data_,target_,rate_,dropout_rate,
    accuracy,optimizer=None,target_hot=None,):
        """
        Starting Training or Testing.
        
        Arguments:
        ----------
            loader(func): load data set,function in LoadData.py, return a generator.
            x(array): data set. the shape is [batch,231,231,3].
            y(array): the true label. shape is [batch,].
            epoch(int): current epoch.
            epochs(int): total epochs.
            mode(str): can choose 'Train' or 'Test'.
            N(int): number of total samples.
            cost(func): loss function.
            placeholder:
                1. data_: data set.
                2. target_: labels.
                3. rate_ : dropout rate.
                4. target_hot: shape is [batch,n_classes], it's a hot matrix.
            dropout_rate(float): feed dropout rate.
            accuracy(tensor): In top-K function.
            optimizer(func): optimizer function,default None like 'Testing' mode.
            
            Returns:
            -------
                acc_(float): mean accuracy rate.
                loss_(float): mean loss value.
        """
        next_element_ = loader.get_batch(x, y)
        acc_, loss_, count, batch_m = 0, 0, 0, 0
        while 1:
            try:
                images, labels = self.sess.run(next_element_)
                hot_labels = np.eye(n_classes)[labels]
                batch_m += images.shape[0]
                print(
                    "[{}/{}] {} [{}/{}]\r".format(epoch + 1, epochs, mode,
                                                  batch_m, N),
                    end="",
                    flush=True,
                )
                if mode == "Train":
                    _, loss = self.sess.run(
                        [optimizer, cost],
                        feed_dict={
                            data_: images,
                            target_hot: hot_labels,
                            rate_: dropout_rate,
                        },
                    )
                    acc = self.sess.run(
                        accuracy,
                        feed_dict={
                            data_: images,
                            target_: labels,
                            rate_: dropout_rate
                        },
                    )
                else:
                    acc, loss = self.sess.run(
                        [accuracy, cost],
                        feed_dict={
                            data_: images,
                            target_: labels,
                            target_hot: hot_labels,
                            rate_: dropout_rate,
                        },
                    )

                acc_ += acc
                loss_ += loss
                count += 1
            except tf.errors.OutOfRangeError:
                acc_ /= count
                loss_ /= count
                return acc_, loss_

    def running(self, lr, epochs, dropout_rate, top_k=5, save_model=True):
        """
        Start build OverFeat model.
        -----------------------------------------------------------------------------------------------------------------
         This function have 7 part.
         1. Create placeholder: data_;target_hot(using cost func);target_(using top-k func);rate_(using forward func).
         2. Initialization parameters.
         3. Caclulate Top-K accuracy.
         4. Load Data set. The class in LoadData.py.
         5. Build Saver to save trained model,just have weighat and bias.
         6. Start training or testing.
         7. Save Model
        ----------------------------------------------------------------------------------------------------------------

        Arguments:
        ---------
            lr(float): learning rate.
            epochs(int): training epochs.
            dropout_rate(float): dropout rate.
            top_k(int): top-k accuracy. defaule 5.
            save_model(bool): save model,default True.
        """
        # placeholder
        data_ = tf.placeholder(tf.float32,
                               (None, resize_image[0], resize_image[1], 3),
                               "Input")
        target_hot = tf.placeholder(tf.float32, (None, n_classes),
                                    "Target_hot")
        target_ = tf.placeholder(tf.int32, (None), "Target_")
        rate_ = tf.placeholder(tf.float32, name="DropoutRate")

        # init parameters.
        params = self.init_params()
        out = self.forward(data_, params, rate_)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_hot,
                                                       logits=out))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)

        # Top_K of accuracy
        accuracy = self.top_k(out, target_, top_k)

        # Load Data
        split_data = SplitDataset(test_rate=self.test_rate)
        sample_train, sample_test = split_data.load_files()
        Images_train, Labels_train = sample_train
        Images_test, Labels_test = sample_test
        N_train = Images_train.shape[0]
        N_test = Images_test.shape[0]
        loader = LoadData(self.sess, batch_size=self.batch_size)

        init = tf.global_variables_initializer()
        # Saver..
        saver = tf.train.Saver(var_list=params)
        
        # Training...
        self.sess.run(init)
        for epoch in range(epochs):
            # Train
            acc_train, loss_train = self.Go(
                loader=loader,
                x=Images_train,
                y=Labels_train,
                epoch=epoch,
                epochs=epochs,
                mode="Train",
                N=N_train,
                cost=cost,
                data_=data_,
                target_=target_,
                rate_=rate_,
                dropout_rate=dropout_rate,
                accuracy=accuracy,
                optimizer=optimizer,
                target_hot=target_hot,
            )
            # Test
            acc_test, loss_test = self.Go(
                loader=loader,
                x=Images_test,
                y=Labels_test,
                epoch=epoch,
                epochs=epochs,
                mode="Test",
                N=N_test,
                cost=cost,
                data_=data_,
                target_=target_,
                rate_=rate_,
                dropout_rate=0,
                accuracy=accuracy,
                optimizer=None,
                target_hot=target_hot,
            )
            print("[{}/{}] Train loss:{} acc:{} Test: loss:{} acc:{}".format(
                epoch + 1, epochs, loss_train, acc_train, loss_test, acc_test))
            if acc_train > 0.95:
                break
        if save_model:
            # Saver ....
            save_path = saver.save(self.sess, "../MODELS/OverFeat_C/model.ckpt")
            print("Model saved in path: %s" % save_path)


if __name__ == "__main__":

    overfeat = OverFeat(batch_size=64, test_rate=0.1)
    overfeat.running(lr=1e-4,
                     epochs=30,
                     dropout_rate=0.5,
                     top_k=5,
                     save_model=True)
