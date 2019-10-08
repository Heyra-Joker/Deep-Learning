import numpy as np
import tensorflow as tf


from Loss import _losses
from GetDataInfo import FilesLoader, DataLoad
from Utels import _get_response_obj
from Utels import _get_confidence
from Utels import _get_TP_FPs
from Utels import _get_precision_recall
from Utels import _get_mAP

class YoLoVgg16:
    def __init__(self,VOC_dir,batch_size):
        self.VOC_dir = VOC_dir
        self.batch_size = batch_size
        self.resize_image = 448
        self.map_iou_thresh = 0.3
        self.var_list = []
        self.sess = tf.Session()
        self.w_init = tf.initializers.glorot_normal()
        self.b_init = tf.initializers.zeros()
        self.vgg_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

    def _cal_mAP(self, feed_dict):
        TP_FP, GT_sum = self.sess.run([self.TP_FP, self.GT_sum], feed_dict=feed_dict)
        precisions, recalls = _get_precision_recall(TP_FP, GT_sum)
        mAP = _get_mAP(precisions, recalls)
        return mAP

    def _leak_relu(self, data):
        data = tf.nn.leaky_relu(data, 0.1)
        return data

    def _conv2d(self,data,name,shape,strides):
        weights = tf.get_variable('W'+name, shape, initializer=self.w_init)
        bias = tf.get_variable('b'+name, (1, 1, shape[-1]), initializer=self.b_init)
        self.var_list.append(weights)
        self.var_list.append(bias)
        data = tf.add(tf.nn.conv2d(data, weights, strides, padding='SAME'), bias)
        data = self._leak_relu(data)
        return data

    def _fc_layer(self, data, name, shape, activation=None, drop=False):
        weights = tf.get_variable('W' + name, shape, initializer=self.w_init)
        bias = tf.get_variable('b' + name, (1, shape[-1]), initializer=self.b_init)
        self.var_list.append(weights)
        self.var_list.append(bias)
        data = tf.add(tf.matmul(data, weights), bias)
        if activation:
            data = activation(data)
        if drop:
            data = tf.nn.dropout(data, rate=0.5)
        return data

    def _net(self, data):
        data = self._conv2d(data, '21', (3, 3, 512, 1024), (1, 1, 1, 1))
        data = self._conv2d(data, '22', (3, 3, 1024, 1024), (1, 2, 2, 1))
        data = self._conv2d(data, '23', (3, 3, 1024, 1024), (1, 1, 1, 1))
        data = self._conv2d(data, '24', (3, 3, 1024, 1024), (1, 1, 1, 1))
        data = tf.reshape(data, (-1, 7 * 7 * 1024))
        data = self._fc_layer(data, '25', (7 * 7 * 1024, 4096), self._leak_relu, drop=False)
        data = self._fc_layer(data, '26', (4096, 7 * 7 * 30))
        data = tf.reshape(data, (-1, 7, 7, 30))
        return data
    
    def _load_data_set(self):
        files_loader = FilesLoader(self.VOC_dir)
        sample_train, sample_val = files_loader.get_datas()
        data_load = DataLoad(self.sess, self.batch_size, self.resize_image)
        return data_load,sample_train,sample_val

    def _running(self,samples, loader, mode, optimizer=False):
        (dataset, labels) = samples
        N = dataset.shape[0]
        next_element = loader.get_batchs(dataset,labels)
        mean_map, mean_loss, count, bs = 0, 0, 0, 0
        while True:
            try:
                image, label = self.sess.run(next_element)
                bs += image.shape[0]
                count += 1
                print('Running {} {}-{}...\r'.format(mode, bs, N), end='', flush=True)
                pre_out = self.vgg_model.predict(image)
                feed_dict = {self.datas:pre_out, self.targets:label, self.learning_rate:self.lr}
                if optimizer:
                    self.sess.run(self.Optimizer,feed_dict=feed_dict)
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                # 计算MAP
                mAP = self._cal_mAP(feed_dict)
                mean_map += mAP
                mean_loss += loss
            except tf.errors.OutOfRangeError:
                mean_loss /= count
                mean_map /= count
                return mean_loss,mean_map

    def yolo_go(self, epochs):
        self.datas = tf.placeholder(tf.float32, (None, 14, 14, 512))
        self.targets = tf.placeholder(tf.float32, (None, 7, 7, 25))
        self.learning_rate = tf.placeholder(tf.float32)
        self.out = self._net(self.datas)
        self.response_pre,self.new_labels = _get_response_obj(self.out, self.targets)
        self.C_predict = _get_confidence(self.response_pre)
        self.TP_FP, self.GT_sum = _get_TP_FPs(self.C_predict, self.targets, self.map_iou_thresh)
        self.cost = _losses(self.response_pre, self.new_labels)
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, var_list=self.var_list)

        # 加载数据
        data_load, sample_train, sample_val = self._load_data_set()
        # 保存模型
        saver = tf.train.Saver(self.var_list)

        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs+1):
            if epoch < 50:
                self.lr = 10e-5
            else:
                self.lr = 10e-6
            train_loss,train_map = self._running(sample_train, data_load, 'Trian', optimizer=True)
            val_loss,val_map = self._running(sample_val, data_load, 'Val', optimizer=False)
            print('{}-{} train loss:{:.4f} train map:{:.4f} val loss:{:.4f} val map:{:.4f}'.format(
                epoch, epochs, train_loss,train_map, val_loss, val_map))
            if val_map >=0.5:
                break
        save_path = saver.save(self.sess, "YOLO_VGG16/YOLO_VGG16_model.ckpt")
        print("Model saved in path: %s" % save_path)
        

if __name__ == "__main__":
    VOC_dir = '/Users/joker/jokers/DataSet/VOCdevkit_train/VOC2012'
    yolo_vgg16 = YoLoVgg16(VOC_dir, 64)
    yolo_vgg16.yolo_go(epochs=100)
