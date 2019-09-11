import tensorflow as tf
import numpy as np
import os
from lxml import etree
from Config import resize_image,Annotation_Files,Images_Files


"""
This Code to Split Data set, load data with "Tensorflow Dataset".
-------------------------------------------------------------
https://tensorflow.google.cn/tutorials/load_data/images?hl=en
--------------------------------------------------------------
"""

class SplitDataset:
    """
    Split data set.
    """
    def __init__(self, test_rate=0.2):
        """
        Argument:
        --------
            test_rate(float): test data rate,default 0.2.
        """
        self.FlipClasses = {}
        self.test_rate = test_rate

    def read_classes_(self):
        """
        Read classes in Classes_.txt, and build dictionary of classes_name-index.
        """
        open_ = open("../Classes_.txt", mode="r")
        lines = open_.readlines()
        for line in lines:
            line_ = line.strip()
            index, classes = line_.split()
            self.FlipClasses[classes] = int(index)
        open_.close()


    def shuffle(self, Images, Labels):
        """
        shuffle data and labels.
        Arguments:
        ---------
            Images(array): data files.,the shape is [N,],every param is "image path" in Image array.
            Labels(array): data labels, shape is [N,],like a=[0,3,5,17,20,...] len(a) = N
        Returns:
        --------
            sample_train(tuple): train data files path,and train labels.
            sample_test(tuple): test data files path, and test labels.
        """
        m = Images.shape[0]
        # shuffle array, Images and Labels.
        permutation = np.random.permutation(m)
        Images = Images[permutation]
        Labels = Labels[permutation]


        m_test = int(self.test_rate * m)
        print("[+] Train samples:{} Test samples:{}".format(
            m - m_test, m_test))
        Images_test = Images[:m_test]
        Labels_test = Labels[:m_test]
        

        Images_train = Images[m_test:]
        Labels_train = Labels[m_test:]
        

        sample_train = (Images_train, Labels_train)
        sample_test = (Images_test, Labels_test)

        return sample_train, sample_test

    def load_files(self):
        """
        load file:
        open annotation dir =>    read all dir       ==> read all xml
                                (get classes name)

        """
        # create classes name dir.
        self.read_classes_()

        Images = []
        Labels = []

        # start read xml and cache iamge path in Image list,cache label in Label list.
        annotation_dir = os.listdir(Annotation_Files)
        for a_dir in annotation_dir:
            classes_name = a_dir.split("-")[-1]
            # labels
            index = self.FlipClasses[classes_name]
            # get all xml files.
            a_path = os.path.join(Annotation_Files, a_dir)
            xml_files = os.listdir(a_path)
            for xml in xml_files:
                image_name = xml # the image name is xml name.
                # images
                image_path = os.path.join(Images_Files, a_dir,
                                          image_name + ".jpg")
                Images.append(image_path)
                Labels.append(index)

        Images = np.array(Images)
        Labels = np.array(Labels)

        sample_train, sample_test = self.shuffle(Images, Labels)

        return sample_train, sample_test


class LoadData:
    """
    Using Tensorflow Dataset class to load batch data.
    """
    def __init__(self, sess, batch_size):
        self.batch_size = batch_size
        self.sess = sess

    def _parse(self, features, labels):
        image = tf.read_file(features) # read files
        image = tf.image.decode_jpeg(image, channels=3) # decode jpeg and the channels must be 3.
        image = tf.image.resize(image, resize_image) # resize to target size,defult is 231.
        image = tf.math.divide(image, 255) # ones images.
        return image, labels

    def get_batch(self, features, labels):
        features_ = tf.placeholder(features.dtype, features.shape)
        labels_ = tf.placeholder(labels.dtype, labels.shape)

        data_set = tf.data.Dataset.from_tensor_slices((features_, labels_))
        data_set = data_set.map(self._parse)
        data_set = data_set.shuffle(1000)
        data_set = data_set.batch(self.batch_size)

        iterator_ = data_set.make_initializable_iterator()
        next_element = iterator_.get_next()
        self.sess.run(iterator_.initializer,
                      feed_dict={
                          features_: features,
                          labels_: labels
                      })
        return next_element


if __name__ == "__main__":

    from PIL import Image, ImageDraw

    sess = tf.Session()
    split_data = SplitDataset(test_rate=0.2)
    sample_train, sample_test = split_data.load_files()
    # train
    Images, Labels = sample_train
    print(Images.shape)
    print(Labels.shape)

    loader = LoadData(sess, batch_size=100)
    for epoch in range(5):
        next_element_train = loader.get_batch(Images,Labels)
        x, y = sess.run(next_element_train)
        print(y[0])
        img = Image.fromarray((x[0] * 255).astype("uint8"))
        img.show()
