import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16


from Classes import classes_name
from Load_Flies import LoadFiles_
from LoadDataset import Loader
"""
#############################
# Classification of VGG-16  #
#############################

The Paper address:
    https://arxiv.org/pdf/1409.1556.pdf

1. In the code, using VGG-16 to build Classification.
2. Since the model is too large, we didn’t train the model from scratch but use transfer learning.
3. It uses Keras vgg-16 model and downloads weights file in "~/.keras/models/ ".
4. It implemented three fully connected layers by "FC14,FC15,FC16".

Note:
-----
	1. Because the data set is an uneven distribution of categories, so the model may be overfitted.
	2. Shallow FC layers do not use dropout (rate=0) in training.

"""
class C_VGG16:
    def __init__(self, batch_size, Annotation_dir, test_file_save_path, file_split_rate = (0.7,0.2,0.1),model_save_path=None):
        """
        Implementation classification of vgg16.
        Arguments:
        ----------
            batch_size(int): training batch size.
            Annotation_dir(str): xml dir. 
            test_file_save_path(str): test file save path (xx.npy).
            file_split_rate(tuple):split rate of data set,(train,val,test),default (0.7,0.2,0.1).
            model_save_path:save trained model.default None.
        """
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.Annotation_dir = Annotation_dir
        self.n_classes = len(classes_name)
        self.file_split_rate = file_split_rate
        self.test_file_save_path = test_file_save_path
        self.model_save_path = model_save_path
        self.vgg_model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))

    def Keras_Vgg16(self, data):
        """
        Pre-train vgg16, the layer incloud (C1-C13).
        Argments:
        ---------
            data(tensor): images, the shape like (bs,h,w,c)
        Returns:
        -------
            features(tensor): return C13 layer output, also like shape (bs,7,7,512).
        """
        features = self.vgg_model.predict(data)
        return features

    def Top_K(self, out, labels, k):
        """
        calculate top-k at model.
        Arguments:
        ---------
            out(tensor): predict values. shape (bs,n_classes)
            labels(tensor): true labels. shape equal out.
            k(int): choose K accuracy.
        Return:
        ------
            accuracy(float): top-k accuracy in current batch size.
        """
        softmax_ = tf.nn.softmax(out, axis=1)
        labels = tf.argmax(labels, axis=1)
        in_top_k = tf.math.in_top_k(softmax_, labels, k)
        accuracy = tf.reduce_mean(tf.cast(in_top_k, tf.float32))
        return accuracy
    
    def LoadDataset(self):
        """
        Load Data set by stanford-dogs.
        More information,please view LoadDataset.py
        """
        loadfiles_ = LoadFiles_(self.Annotation_dir,self.file_split_rate,
        self.test_file_save_path,target_mode='labels')

        samples_train,samples_val = loadfiles_.load_files()

        self.N_train = samples_train[0].shape[0]
        self.N_val = samples_val[0].shape[0]
        loader = Loader(self.sess, batch_size=self.batch_size)
        return loader,samples_train,samples_val
    
    def Running(self, epoch, epochs, N, loader, data_mode, samples, lr, 
                features, targets, learning_rate, drop_rate, cost, accuracy, optimizer=None):
        """
        Start running model(train/val).
        Argumens:
        --------
            Tensorflow Placeholder(tensor):
            ==============================================================
            1.features: input layer, and shape same as (None, 7, 7, 512)
            2.targets: True labels(Targets). (None, n_classes)
            3.learning_rate: learning rate in Backward Propagation.
            4.drop_rate: dropout layer parameter.
            ==============================================================
            
            Tensorflow Cost and Optimizer:
            ================================================================================
            1.cost: loss function,use softmax-logist.
            2.optimizer: optimizer function(Adam),default None.In val model,it can set None.
            =================================================================================
            
            Other Params:
            =============
            1.epoch/epocs(int): running epoch.
            2.N(int): number of running data.
            3.loader(tensor): generator of tensorflow.
            4.data_mode(str): can choose "train/val".
            5.samples(array): incloud features and labels.train or val data set.
            6.lr: learning rate, when accuracy of val data not changed, lr / 2.
        """
        count, mean_acc, mean_loss, bs = 0, 0, 0, 0
        sample_imgs, sample_labels = samples[0], samples[1]

        next_element = loader.get_batch(sample_imgs, sample_labels, data_mode, bbox=False)

        while True:
            try:
                imgs, labels = self.sess.run(next_element)
                count, bs =  count + 1, bs + imgs.shape[0]
                print('[+] EPOCHS:{}-{} {}: [{}-{}]\r'.format(epoch, epochs, data_mode, bs, N),end='',flush=True)
                pre_imgs = self.Keras_Vgg16(imgs)
                # handel feed_dict, In this code(dataset), need set dropout rate equal zero.
                feed_dict = {features:pre_imgs, targets:labels, learning_rate:lr,drop_rate:0}
                # In train, then run optimizer.
                if optimizer:
                    self.sess.run(optimizer, feed_dict=feed_dict)
                # get acc and loss
                loss, acc = self.sess.run([cost, accuracy], feed_dict=feed_dict)
                mean_loss, mean_acc = mean_loss + loss, mean_acc + acc

            except tf.errors.OutOfRangeError:
                mean_loss, mean_acc = mean_loss / count, mean_acc / count
                return mean_loss, mean_acc
        
    def Init_params(self):
        """
        Init parameters.Just have three layers:
        1.FC-14
        2.FC-15
        3.FC-16(out layer)
        """
        self.ParmsDict = {}
        init_weights = tf.initializers.glorot_normal()
        init_bias = tf.initializers.zeros()
        # Fully Connect
        self.ParmsDict['W14'] = tf.get_variable('W14', (7 * 7 * 512, 4096),
                                                initializer=init_weights)
        self.ParmsDict['b14'] = tf.get_variable('b14', (1, 4096), 
                                                initializer=init_bias)
        self.ParmsDict['W15'] = tf.get_variable('W15', (4096, 4096),
                                                initializer=init_weights)
        self.ParmsDict['b15'] = tf.get_variable('b15', (1, 4096), 
                                                initializer=init_bias)
        self.ParmsDict['W16'] = tf.get_variable('W16', (4096, self.n_classes), 
                                                initializer=init_weights)
        self.ParmsDict['b16'] = tf.get_variable('b16', (1, self.n_classes), 
                                                initializer=init_bias)
        
    def FC(self, data, W, b):
        """
        Fully Connect Layer.
        """
        F = tf.add(tf.matmul(data, W), b)
        return F

    def Net(self, data, rate):
        """
        Transfer Model.
        """
        data = tf.reshape(data, (-1, 7 * 7 * 512))
        # FC14
        F14 = self.FC(data, self.ParmsDict['W14'], self.ParmsDict['b14'])
        R14 = tf.nn.relu(F14)
        D14 = tf.nn.dropout(R14,rate=rate)
        # FC15
        F15 = self.FC(D14, self.ParmsDict['W15'], self.ParmsDict['b15'])
        R15 = tf.nn.relu(F15)
        D15 = tf.nn.dropout(R15,rate=rate)
        # FC16
        Out = self.FC(D15, self.ParmsDict['W16'], self.ParmsDict['b16'])
        return Out

    def Vgg(self, lr, epochs, k=5):
        """
        Build Transfer of VGG-16 model.

        Arguments:
        ---------
            lr: learning rate.
            epochs: training epochs.
            k: top-K parameter, default None.
        """
        # Create placeholder.
        features = tf.placeholder(tf.float32, (None, 7, 7, 512), name='Input')
        targets = tf.placeholder(tf.float32, (None, self.n_classes), name='Targets')
        learning_rate = tf.placeholder(tf.float32, name='Learning_rate')
        drop_rate = tf.placeholder(tf.float32, name='Drop_rate')

        # Initilization params
        self.Init_params()
        
        # forward
        OUT = self.Net(features,drop_rate)
        COST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=OUT,labels=targets))
        OPTIMIZER = tf.train.AdamOptimizer(learning_rate).minimize(COST)
        
        # accuracy
        ACCURACY = self.Top_K(OUT, targets, k)

        # Load Dataset
        loader,samples_train,samples_val = self.LoadDataset()

        # start training...
        self.sess.run(tf.global_variables_initializer())
        # saver
        saver = tf.train.Saver([weight for weight in self.ParmsDict.values()])
        
        Val_Accuracy = 0
        # Start training model.
        for epoch in range(1, epochs + 1):
            # train
            train_loss, train_acc = self.Running(epoch, epochs, self.N_train,loader, 
                                                'train', samples_train, lr, features, 
                                                targets, learning_rate, drop_rate, COST, 
                                                ACCURACY, optimizer=OPTIMIZER)
            # val
            val_loss, val_acc = self.Running(epoch, epochs, self.N_val,loader, 
                                                'val', samples_val, lr, features, 
                                                targets, learning_rate, drop_rate, COST, 
                                                ACCURACY, optimizer=None)
            print('[+] {}-{}: train loss:{:.4f} train acc:{:.4f} val loss:{:.4f} val acc:{:.4f}'.format(
                epoch, epochs, train_loss, train_acc, val_loss, val_acc
            ))

            if val_acc - Val_Accuracy <= 1e-5:
                lr /= 2
            Val_Accuracy = val_acc

            # Saver ....
            if self.model_save_path:
                if val_acc >= 0.6 and train_acc >= 0.8:
                    save_path = saver.save(self.sess, self.model_save_path)
                    print("Model saved in path: %s" % save_path)
                    break
    

if __name__ == "__main__":
    Annotation_dir = '/Users/joker/jokers/DataSet/stanford-dogs-dataset/Annotation'
    test_file_save_path = 'TEST_FILES'
    model_save_path = 'ModelCVgg16/C_VGG16.ckpt'
    c_vgg16 = C_VGG16(batch_size=64, Annotation_dir=Annotation_dir, 
    test_file_save_path=test_file_save_path,model_save_path=model_save_path)
    c_vgg16.Vgg(lr=0.0001, epochs=50, k=1)
   


        


