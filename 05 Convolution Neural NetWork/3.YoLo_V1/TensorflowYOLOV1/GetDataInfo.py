import os
import time
import json
import numpy as np
from lxml import etree
import tensorflow as tf
from cv2 import cv2
from Utels import _get_cell_info

class FilesLoader:
    def __init__(self, VOC_dir, val_rate=0.3, save_path=None):
        self.VOC_dir = VOC_dir
        self.val_rate = val_rate

        self.save_path = save_path
        self.resize_wh = 448
        self.cells_num = 7 # 分成7格
        self.Annotations_path = os.path.join(self.VOC_dir, 'Annotations')
        self.JPEGImages = os.path.join(self.VOC_dir, 'JPEGImages')
        self.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

    def _save_classes_id(self, classes_id):
        """保存标签信息"""
        path = 'flip_classes_id.json'
        if self.save_path:
            path = self.save_path
        with open(path, 'w', encoding='utf8') as file:
            file.write(json.dumps(classes_id, indent=2, ensure_ascii=False))
        print('[+] The classes_id file save in %s'%path)

    def _get_classes_id(self):
        """获取标签的id"""
        classes_id = dict(zip(self.CLASSES, range(len(self.CLASSES))))
        flip_classes_id = dict(zip(range(len(self.CLASSES)),self.CLASSES))
        return classes_id, flip_classes_id
    
    def _get_image_wh(self,tree):
        """获取图片宽高"""
        width = int(tree.xpath('//annotation/size/width/text()')[0])
        height = int(tree.xpath('//annotation/size/height/text()')[0])
        return width, height
    
    def _get_image_path(self, tree):
        """获取图片路径"""
        img_name = str(tree.xpath('//annotation/filename/text()')[0])
        img_path = os.path.join(self.JPEGImages, img_name)
        return img_path

    def _get_label_info(self, tree):
        """获取标签信息"""

        # 获取图片原始宽高
        width, height = self._get_image_wh(tree)
        objects = tree.xpath('//annotation/object')
        label = np.zeros((7,7,25)) # label排列符合:[1-confidences, 4-bboexs,20-classes]
        for object in objects:
            name = str(object.xpath('./name/text()')[0].strip())
            # 获取对应标签索引
            cls_id = self.classes_id[name]
            # 获取坐标,存在精度丢失问题,转换成int
            xmin = int(float(object.xpath('./bndbox/xmin/text()')[0]))
            ymin = int(float(object.xpath('./bndbox/ymin/text()')[0]))
            xmax = int(float(object.xpath('./bndbox/xmax/text()')[0]))
            ymax = int(float(object.xpath('./bndbox/ymax/text()')[0]))
            # 处理cell信息
            location_list = (xmin, ymin, xmax, ymax)
            x_id, y_id, bboxes = _get_cell_info(width,height,self.resize_wh,location_list,self.cells_num)
            
            if label[x_id,y_id,0] == 1:
                continue
            else:
                # [c, x, y, w, h, classes]
                label[x_id,y_id,0] = 1 
                label[x_id,y_id,1:5] = bboxes
                label[x_id,y_id,5 + cls_id] = 1
        return label
            
    def _get_annotations(self):
        """获取文件所有需要信息"""
        LABELS = []
        IMAGES = []
        anno_fiels = os.listdir(self.Annotations_path)
        for file in anno_fiels:
            a_files_path = os.path.join(self.Annotations_path, file)
            tree = etree.parse(a_files_path)
            # 获取图片路径
            img_path = self._get_image_path(tree)
            # 获取标签信息
            label = self._get_label_info(tree)
            # 添加 image path
            IMAGES.append(img_path)
            # 添加 label
            LABELS.append(label)
        IMAGES = np.array(IMAGES)
        LABELS = np.array(LABELS)
        return IMAGES,LABELS
    
    def _shuffle(self, data, labels):
        m = data.shape[0]
        N_val = int(self.val_rate * m)
        # 打乱数据集
        permutation = np.random.permutation(m)
        data = data[permutation,...]
        labels = labels[permutation,...]
        # 划分验证集
        data_val = data[:N_val,...]
        labels_val = labels[:N_val,...]
        # 划分训练集
        data_train = data[N_val:,...]
        labels_train = labels[N_val:,...]

        sample_val = (data_val[:2], labels_val[:2])
        sample_train = (data_train[:2], labels_train[:2])
        print(data_train[:2])
        print('[+] train samples:{} Val samples:{}'.format(data_train.shape[0], data_val.shape[0]))
        return sample_train, sample_val

    def get_datas(self):
        """启动函数"""
        self.classes_id,flip_classes_id = self._get_classes_id()
        # 保存classes_id作为测试时候使用,{0:person,1:airplane}...
        self._save_classes_id(flip_classes_id)
        print('[*] start handel VOC dataset...')
        start_time = time.time()
        IMAGES,LABELS = self._get_annotations()
        sample_train, sample_val = self._shuffle(IMAGES,LABELS)
        end_time = time.time()
        print('[+] Handel VOC dataset done! used time:%.4f sec.'%(end_time - start_time))
        return sample_train, sample_val

class DataLoad:
    def __init__(self, sess, batch_size, image_resize):
        self.sess = sess
        self.batch_size = batch_size
        self.image_resize = image_resize
    def __paser(self, image, label):
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (self.image_resize, self.image_resize))
        image = tf.math.divide(image, 255)
        return image, label

    def get_batchs(self, datas, labels):
        features = tf.placeholder(datas.dtype, datas.shape)
        targets = tf.placeholder(labels.dtype, labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        dataset = dataset.map(self.__paser)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        self.sess.run(iterator.initializer, feed_dict={features:datas, targets:labels})
        return next_element

if __name__ == "__main__":
    from Show import FileShow
    VOC_dir = '/Users/joker/jokers/DataSet/VOCdevkit_train/VOC2012'
    files_loader = FilesLoader(VOC_dir)
    sample_train, sample_val = files_loader.get_datas()
    (data_train, labels_train) = sample_train
    print(data_train.shape)
    print(labels_train.shape)
    # 查看原图是否能够显示
    for i in range(5):
        FileShow(data_train[i], labels_train[i])
    # sess = tf.Session()
    # data_load = DataLoad(sess, 64, 448)
    # next_element = data_load.get_batchs(data_train,labels_train)
    # while True:
    #     try:
            
    #         imgs,labels = sess.run(next_element)
    #         # test_image = (imgs[0]* 255).astype('uint8')
    #         # test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    #         # cv2.imshow('img',test_image)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
            
    #     except tf.errors.OutOfRangeError:
    #         break
