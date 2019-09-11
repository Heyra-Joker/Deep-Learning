import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16


from Load_Flies import LoadFiles_
from LoadDataset import Loader


"""
#############################
#  Regression of VGG-16     #
#############################

The Paper address (A LOCALISATION):
    https://arxiv.org/pdf/1409.1556.pdf

1. Since the model is too large, we didn’t train the model from scratch but use transfer learning.
2. It uses Keras vgg-16 model and downloads weights file in "~/.keras/models/ ".
3. It implemented three fully connected layers by "FC14,FC15,FC16".

Note:
----
    In this code, we used (per-class regression, PCR), course you can build(single-class regression, SCR).
"""
class R_VGG16:
    def __init__(self, batch_size, Annotation_dir, test_file_save_path, bbox_classes, split_rate=(0.7,0.2,0.1), model_save_path=None):
        """
        batch_size(int): batch size.
        Annotation_dir(string): xml dir. 
        test_file_save_path(string): test file save path (xx.npy).
        bbox_classes(string): target classes name in regression VGG.
        split_rate(tuple):split rate of data set,(train,val,test),default (0.7,0.2,0.1).
        model_save_path(tsring): save model path, default None.
        """
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.Annotation_dir = Annotation_dir
        self.test_file_save_path = test_file_save_path
        self.bbox_classes = bbox_classes
        self.split_rate = split_rate
        self.model_save_path = model_save_path

        self.n_classes = 4
        self.vgg_model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))

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
        predict_box = np.exp(predict_box)
        true_box = np.exp(true_box)
        in_w = np.minimum(predict_box[:, 2], true_box[:, 2]) - np.maximum(
            predict_box[:, 0], true_box[:, 0])
        in_h = np.minimum(predict_box[:, 3], true_box[:, 3]) - np.maximum(
            predict_box[:, 1], true_box[:, 1])

        # in_w 和in_h 都大于0 则使用面积,否则为0
        inter = np.logical_and(np.greater(in_w, 0), np.greater(
            in_h, 0))* (in_w * in_h)
        union = np.multiply((predict_box[:,3] - predict_box[:,1]),(predict_box[:,2] - predict_box[:,0])) + \
            np.multiply((true_box[:,3] - true_box[:,1]),(true_box[:,2] - true_box[:,0])) - inter
        iou = np.divide(inter, union)
        return iou
    
    def LoadDataset(self):
        """
        Load Data set by stanford-dogs.
        More information,please view LoadDataset.py
        """
        loadfiles_ = LoadFiles_(self.Annotation_dir,
                                self.split_rate,
                                self.test_file_save_path,
                                target_mode='bboxs',
                                bbox_classes=self.bbox_classes)

        samples_train,samples_val = loadfiles_.load_files()

        self.N_train = samples_train[0].shape[0]
        self.N_val = samples_val[0].shape[0]
        loader = Loader(self.sess, batch_size=self.batch_size)
        return loader,samples_train,samples_val

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
    
    def Running(self,samples, loader, data_mode, epoch, epochs, N,
                features, targets, learning_rate, lr, drop_rate,
                out, cost, iou_thresh , optimizer=None):
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
            7.iou_thresh: calculate iou.
        """
        mean_iou, mean_loss, count, bs = 0, 0, 0, 0
        sample_imgs, sample_labels = samples[0], samples[1]
        next_element = loader.get_batch(sample_imgs, sample_labels, data_mode, bbox=True)

        while True:
            try:
                imgs, labels = self.sess.run(next_element)
                count, bs =  count + 1, bs + imgs.shape[0]
                print('[+] EPOCHS:{}-{} {}: [{}-{}]\r'.format(epoch, epochs, data_mode, bs, N),end='',flush=True)
                pre_imgs = self.Keras_Vgg16(imgs)
                # handel feed_dict, In this code(dataset), need set dropout rate equal zero.
                feed_dict = {features:pre_imgs, targets:labels, learning_rate:lr, drop_rate:0}
                # In train, then run optimizer.
                if optimizer:
                    self.sess.run(optimizer, feed_dict=feed_dict)
                out_, mean_loss = self.sess.run([out,cost], feed_dict=feed_dict)
                iou_value = np.mean(self.Iou(out_, labels))
                mean_iou += iou_value
            except tf.errors.OutOfRangeError:
                print()
                mean_loss, mean_iou = mean_loss / count, mean_iou / count
                return mean_iou, mean_loss
        
    
    def Vgg(self, lr, epochs, iou_thresh=0.5):

        # Create placeholder.
        features = tf.placeholder(tf.float32, (None, 7, 7, 512), name='Input')
        targets = tf.placeholder(tf.float32, (None, self.n_classes), name='Targets')
        learning_rate = tf.placeholder(tf.float32, name='Learning_rate')
        drop_rate = tf.placeholder(tf.float32, name='Drop_rate')

        # Initilization params
        self.Init_params()
        
        # forward
        OUT = self.Net(features,drop_rate)
        COST = tf.reduce_mean(tf.losses.huber_loss(targets, OUT))
        OPTIMIZER = tf.train.AdamOptimizer(learning_rate).minimize(COST)

        # Load Dataset
        loader,samples_train,samples_val = self.LoadDataset()
        
        # init global variables.
        self.sess.run(tf.global_variables_initializer())
        
        # saver
        saver = tf.train.Saver([weight for weight in self.ParmsDict.values()])

        
        for epoch in range(1, epochs+1):
            # train
            iou_train, loss_train = self.Running(samples=samples_train, 
                                                loader=loader, 
                                                data_mode='train', 
                                                epoch=epoch, 
                                                epochs=epochs,
                                                N=self.N_train,
                                                features=features,
                                                targets=targets,
                                                learning_rate=learning_rate,
                                                lr=lr,
                                                drop_rate=drop_rate,
                                                out=OUT,
                                                cost=COST,
                                                iou_thresh=iou_thresh,optimizer=OPTIMIZER)
            # val
            iou_val, loss_val = self.Running(   samples=samples_val, 
                                                loader=loader, 
                                                data_mode='val', 
                                                epoch=epoch, 
                                                epochs=epochs,
                                                N=self.N_val,
                                                features=features,
                                                targets=targets,
                                                learning_rate=learning_rate,
                                                lr=lr,
                                                drop_rate=drop_rate,
                                                out=OUT,
                                                cost=COST,
                                                iou_thresh=iou_thresh,optimizer=None)

            print('[+] {}-{}: train loss:{:.4f} train iou:{:.4f} val loss:{:.4f} val iou:{:.4f}'.format(
                epoch, epochs, loss_train, iou_train, loss_val, iou_val
            ))
           
            # Saver ....
            if self.model_save_path:
                if iou_val >iou_thresh:
                    save_path = saver.save(self.sess, self.model_save_path)
                    print("Model saved in path: %s" % save_path)
                    break



if __name__ == "__main__":

    Annotation_dir = '/Users/joker/jokers/DataSet/stanford-dogs-dataset/Annotation'
    test_file_save_path = 'TEST_FILES'
    bbox_classes = 'n02116738-African_hunting_dog'
    model_save_path = 'ModelRVgg16/R_VGG16.ckpt'
    r_vgg16 = R_VGG16(10, Annotation_dir, test_file_save_path, bbox_classes, model_save_path=model_save_path)
    r_vgg16.Vgg(lr=0.0001, epochs=40, iou_thresh=0.6)
    

