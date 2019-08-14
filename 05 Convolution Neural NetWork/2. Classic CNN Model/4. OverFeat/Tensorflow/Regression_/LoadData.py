import os
import tensorflow as tf
import numpy as np
from lxml import etree
from PIL import Image,ImageDraw
from Config import resize_image, Annotation_Files, Images_Files


"""
Load Data: Images and Bbox.
Note:
-------------------------------------------------------------------------------------
Because we input image size is (231,231), so the box must be equivalent change value.
Like,
The original width=500, height=500, resize_width = 231, resize_height = 231, than,
N = width / resize_width,
bbox_x(xmin,xmax) = original_x / N
the height is so true.
"""

class LoadFiles:
    def __init__(self, load_limit=None, test_rate=0.2):
        """
        Arguments:
        ---------
            load_limit: number of load classes label,if it equal None means load All classes.
            test_rate: test data rate,default 0.2.
        """
        self.annotation_path = Annotation_Files
        self.jpg_image_path = Images_Files
        self.load_limit = load_limit
        self.test_rate = test_rate

    def loadfiles(self):
        """
        load classes and print current classes. 
        """
        a_files = os.listdir(self.annotation_path)
        if self.load_limit:
            a_files = a_files[:self.load_limit]
            load_classes = [f.split('-')[-1] for f in a_files]
            print('Load Classes is:\n %s' % load_classes)
        return a_files

    def split_train_test(self, xml_files):

        # shuffle files
        np.random.shuffle(xml_files)
        N_test = int(len(xml_files) * self.test_rate)
        tests = xml_files[:N_test]
        trains = xml_files[N_test:]
        return trains, tests

    def getbbox(self, tree):
        """
        Get bbox equivalent value.

        Argument:
        --------
            tree(object): lxml object.
        """
        origin_height = np.float(
            tree.xpath("//annotation/size/height/text()")[0])
        origin_width = np.float(
            tree.xpath("//annotation/size/width/text()")[0])

        origin_xmin = np.float(
            tree.xpath("//annotation/object/bndbox/xmin/text()")[0])
        origin_ymin = np.float(
            tree.xpath("//annotation/object/bndbox/ymin/text()")[0])
        origin_xmax = np.float(
            tree.xpath("//annotation/object/bndbox/xmax/text()")[0])
        origin_ymax = np.float(
            tree.xpath("//annotation/object/bndbox/ymax/text()")[0])

        N_height, N_width = (
            origin_height / resize_image[0],
            origin_width / resize_image[1],
        )
        xmin, ymin = origin_xmin / N_width, origin_ymin / N_height
        xmax, ymax = origin_xmax / N_width, origin_ymax / N_height
        bboxs = (xmin, ymin, xmax, ymax)
        
        bboxs = (xmin,ymin,xmax,ymax)
        return bboxs

    def load_imformation(self, xml_paths, files_list):
        """
        get imformation, like data files and bbox.

        Arguments:
        ----------
            xml_paths(str): xml path, can get xmin,ymin,xmax,ymax.
            files_list(str): image file name.
        Return:
        -------
            SAMPLE: incloud IMAGES and BBOXS. the IMAHES is file path. The BBOXS is [xmin,ymin,xmax,ymax]
        """
        jpg_image_dir = os.path.join(self.jpg_image_path,
                                     xml_paths.split('/')[-1])
        IMAGES = []
        BBOXS = []
        for file in files_list:
            image_path = os.path.join(jpg_image_dir, file + '.jpg')
            xml_path = os.path.join(xml_paths, file)
            tree = etree.parse(xml_path)
            bboxs = self.getbbox(tree)
            IMAGES.append(image_path)
            BBOXS.append(bboxs)
        SAMPLE = (np.array(IMAGES), np.array(BBOXS))
        return SAMPLE

    def loader(self):
        """
        loader classe of regression.
        Note:
        ----
            This function is generator can next SAMPLES(train/text) in currect classes.
        """
        a_files = self.loadfiles()
        for dir_ in a_files:

            xml_paths = os.path.join(self.annotation_path, dir_)
            xml_files = os.listdir(xml_paths)
            trains, tests = self.split_train_test(xml_files)
            _classes_name = dir_.split('-')[-1]
            print('Current Classes: {} Train:({}) Test:({})'.format(
                _classes_name, len(trains), len(tests)))
            SAMPLE_TRAIN = self.load_imformation(xml_paths, trains)
            SAMPLE_TEST = self.load_imformation(xml_paths, tests)
            yield SAMPLE_TRAIN, SAMPLE_TEST,_classes_name


class LoadData:
    """
    Using Tensorflow "Dataset"  to get batch data set, and return generator "next_element".
    """
    def __init__(self,sess,batch_size):
        self.sess = sess
        self.batch_size = batch_size
    def _parse(self,feature,label):
        image = tf.read_file(feature)
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.resize(image, resize_image)
        image = tf.math.divide(image,255)
        return image,label

    def get_batchs(self,features,labels):
        _features = tf.placeholder(features.dtype,features.shape,'features')
        _labels = tf.placeholder(labels.dtype,labels.shape,'labels')

        data_set = tf.data.Dataset.from_tensor_slices((_features,_labels))
        data_set = data_set.map(self._parse)
        data_set = data_set.shuffle(1000)
        data_set = data_set.batch(self.batch_size)

        iterator_ = data_set.make_initializable_iterator()
        next_element = iterator_.get_next()

        self.sess.run(iterator_.initializer,feed_dict={_features:features,_labels:labels})

        return next_element

if __name__ == "__main__":
    loader_F = LoadFiles(load_limit=2, test_rate=0.2)
    samples = loader_F.loader()
    sess = tf.Session()
    loader_D = LoadData(sess,64)
    for i in range(2):
        SAMPLE_TRAIN, SAMPLE_TEST,_ = next(samples)
        next_element = loader_D.get_batchs(SAMPLE_TRAIN[0],SAMPLE_TRAIN[1])
        while True:
            try:
                IMAGES,LABELS= sess.run(next_element)
                img = Image.fromarray((IMAGES[0]* 255).astype('uint8'))
                draw = ImageDraw.Draw(img)
                xmin, ymin, xmax, ymax = LABELS[0][0], LABELS[0][1], LABELS[0][2], LABELS[0][3]
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red",width=3)
                img.show()
            except tf.errors.OutOfRangeError:
                break
        
            
