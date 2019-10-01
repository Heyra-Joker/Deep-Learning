import json
import time
import tensorflow as tf
from cv2 import cv2 as cv

from Utels import _NMS
from Show import PredictShow,read_json,CreameShow

class YoloPredict:
    def __init__(self,json_path, image_path, model_path):
        self.sess = tf.Session()
        self.json_path = json_path
        self.image_path = image_path
        self.model_path = model_path
        self.threshold = 0.8 # 类别置信度阈值
        self.iou_thresh = 0.6 # NMS的iou置信度
        self.max_output_size = 10 # 最大输出个数
        self.vgg_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

    def _read_file(self):
        image = tf.read_file(self.image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (448, 448))
        image = tf.math.divide(image, 255)
        image = tf.reshape(image, (1, 448 ,448, 3))
        image = self.sess.run(image)
        return image

    def _leak_relu(self, data):
        data = tf.nn.leaky_relu(data, 0.1)
        return data

    def _conv2d(self, data, w, b, s):
        data = tf.add(tf.nn.conv2d(data, w, s, "SAME"),b)
        data = self._leak_relu(data)
        return data

    def _fc_layer(self, data, w, b, activation=None):
        data = tf.add(tf.matmul(data, w), b)
        if activation:
            data = activation(data)
        return data

    def _load_weights(self):
        W21 = tf.get_variable('W21',(3, 3, 512, 1024))
        b21 = tf.get_variable('b21',(1,1,1024))
        W22 = tf.get_variable('W22',(3, 3, 1024, 1024))
        b22 = tf.get_variable('b22',(1, 1, 1024))
        W23 = tf.get_variable('W23',(3, 3, 1024, 1024))
        b23 = tf.get_variable('b23',(1, 1, 1024))
        W24 = tf.get_variable('W24',(3, 3, 1024, 1024))
        b24 = tf.get_variable('b24',(1, 1, 1024))
        W25 = tf.get_variable('W25',(7 * 7 * 1024, 4096))
        b25 = tf.get_variable('b25',(1, 4096))
        W26 = tf.get_variable('W26',(4096, 7 * 7 * 30))
        b26 = tf.get_variable('b26',(1, 7 * 7 * 30))
        saver = tf.train.Saver([W21,b21,W22,b22,W23,b23,W24,b24,W25,b25,W26,b26])
        # restore target weights and bias.
        saver.restore(self.sess, self.model_path)
        Params = (W21,b21,W22,b22,W23,b23,W24,b24,W25,b25,W26,b26)
        return Params
    
    
    def _net(self, data, params):
        (W21,b21,W22,b22,W23,b23,W24,b24,W25,b25,W26,b26) = params
        data = self._conv2d(data, W21, b21, (1, 1, 1, 1))
        data = self._conv2d(data, W22, b22, (1, 2, 2, 1))
        data = self._conv2d(data, W23, b23, (1, 1, 1, 1))
        data = self._conv2d(data, W24, b24, (1, 1, 1, 1))
        data = tf.reshape(data, (-1, 7 * 7 * 1024))
        data = self._fc_layer(data, W25, b25, activation=self._leak_relu)
        data = self._fc_layer(data, W26, b26, activation=None)
        data = tf.reshape(data, (7, 7, 30))
        return data
    
    def _get_confidence(self,out):
        """获取类别置信度,也就是结果依靠的条件"""
        part_classes = out[:, :, 10:30] # 获取类别概率 [7,7,20]
        part_c1 = tf.expand_dims(out[:, :, 0], axis=-1) # 获取每一个cell的第一个置信度
        part_c2 = tf.expand_dims(out[:, :, 5], axis=-1) # 获取每一个cell的第二个置信度
        part_c = tf.concat([part_c1, part_c2], axis=-1) # 拼接得到[7,7,2]
        # 计算类别置信度
        part_classes = tf.expand_dims(part_classes, axis=2) # [7,7,1,20]
        part_c = tf.expand_dims(part_c, axis=-1) # [7,7,2,1]
        out = tf.multiply(part_c, part_classes) # [7,7,2,20] 表示7x7个cell中每一个cell有2个类别置信度(20).
        return out
    
    def _filter_confidence(self,out):
        """过滤较小的类别置信度"""
        # 筛选两个bbox中较大的置信度,因为一个cell只能是预测一个类别
        reduce_max = tf.reduce_max(out, axis=-1)
        # 筛选出置信度大于阈值的,返回索引[x_id,y_id,c_id],接着依照这个id获取网络输出值使用NMS.
        where = tf.where(tf.greater(reduce_max, self.threshold))
        return where

    def _cal_NMS(self, where, net_out, c_out):
        """
        计算nms
        where:[x_id,y_id,c_id]
        net_out:[7,7,30]
        c_out:[7,7,2,20],2:c1,c2
        """
        # 将net_out中的x,y,w,h拼接到c_out中,即可得到所有需要的信息[7,7,2,25](25:20+5(cxywh),计算NMS是需要xywh,c)
        # 再依照where将目标获取出来进行NMS
        bbox_xywh1 = tf.expand_dims(net_out[:,:,0:5], axis=2)
        bbox_xywh2 = tf.expand_dims(net_out[:,:,5:10], axis=2)
        bbox_xywh = tf.concat([bbox_xywh1, bbox_xywh2], axis=2) # 这里的拼接一定要和c1,c2一致,[7,7,2,4]
        c_out = tf.concat([c_out, bbox_xywh],axis=-1)
        where_out = tf.gather_nd(c_out, where)
        confidence = tf.reduce_max(where_out[:,0:20],axis=1) # NMS传入的应该是20个类别中最大的置信度
        bboxes = where_out[:,21:25] # [x,y,w,h]
        select_index = _NMS(bboxes, confidence, self.iou_thresh, self.max_output_size)
        select_index = tf.expand_dims(select_index, axis=-1)
        pre_where = tf.gather_nd(where, select_index)
        pre_dict = tf.gather_nd(c_out, pre_where)
        pre_where = pre_where[:,:2] # where中只要x_id,y_id
        return pre_where, pre_dict

    def _pre_running(self):
        classes = read_json(self.json_path)
        image = self._read_file()
        pre_image = self.vgg_model.predict(image)
        return pre_image, classes
    
    def PredictImage(self):
        pre_image, classes = self._pre_running()
        data = tf.placeholder(tf.float32, (1, 14, 14, 512))
        Params = self._load_weights()
        net_out = self._net(data, Params)
        c_out = self._get_confidence(net_out)
        where = self._filter_confidence(c_out)
        select_index= self._cal_NMS(where, net_out, c_out)
        start_time = time.time()
        pre_where, pre_dict = self.sess.run(select_index, feed_dict={data:pre_image})
        end_time = time.time()
        print('[*] Use: %.4f ms.'%((end_time - start_time) * 1000))
        PredictShow(image_path=self.image_path , out=pre_dict, index=pre_where, classes=classes)

    def PredictCreame(self):
        classes = read_json(self.json_path)
        data = tf.placeholder(tf.float32, (1, 14, 14, 512))
        Params = self._load_weights()
        net_out = self._net(data, Params)
        c_out = self._get_confidence(net_out)
        where = self._filter_confidence(c_out)
        select_index= self._cal_NMS(where, net_out, c_out)
        cap = cv.VideoCapture(0)
        
        while True:
            _, frame = cap.read()
            if frame is not None:
                image = cv.resize(frame, (448,448))
                rimage = image.reshape((1, 448,448, 3))
                start_time = time.time()
                pre_image = self.vgg_model.predict(rimage)
                pre_where, pre_dict = self.sess.run(select_index, feed_dict={data:pre_image})
                end_time = time.time()
                print('[*] Use: %.4f ms.'%((end_time - start_time) * 1000))
                frame = CreameShow(frame , pre_dict, pre_where, classes,)
                cv.imshow('image', frame)
                if cv.waitKey(1) == ord('q'):
                    break
        cv.destroyAllWindows()
    

if __name__ == "__main__":
    json_path = 'flip_classes_id.json'
    test_image = '../Testimgs/2009_001008.jpg'
    model_path = '../YOLO_VGG16_v1/YOLO_VGG16_model.ckpt'
    yolo_predict = YoloPredict(json_path,test_image,model_path)
    yolo_predict.PredictImage()