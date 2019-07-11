import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class SplitData:
    """
    Split training data and testing data.
    """
    def __init__(self, file_dir, Load_samples=None, Shuffle=True, test_rate=0.3):
        """
        :param file_dir (string): data dir. like '../train' it's a directory.
        :param Load_samples (int,None): loading Number,if equal None loading all files,default None.
        :param Shuffle (bool): is shuffle original files. suggest True.
        :param test_rate (float): test data rate.
        """
        self.file_dir = file_dir
        self.test_rate = test_rate
        self.Load_samples = Load_samples
        self.Shuffle = Shuffle

    def __call__(self):
        """
        :return train_files, test_files (list): include train files path and test files path.
        """
        # loading all path in current file dir.
        files_list = os.listdir(self.file_dir)

        if self.Shuffle:
            np.random.shuffle(files_list)

        if self.Load_samples is not None:
            files_list = files_list[:self.Load_samples]

        # split data
        len_ = len(files_list)
        test_index = int(np.floor(len_ * self.test_rate))
        test_files = files_list[:test_index]
        train_files = files_list[test_index:]

        # join path
        test_files = [os.path.join(self.file_dir, file) for file in test_files]
        train_files = [os.path.join(self.file_dir, file) for file in train_files]

        train_samples, test_samples = len_ - test_index, test_index
        print('Training Samples:{},Testing Samples:{}'.format(train_samples, test_samples))

        return train_files, test_files


class LoadData:
    """
    Load data set.
    have four function:
        1. add_labels: add data label and change input value to nd_array.
        2. parse_images: parse images.
        3. get_batch: create tensorflow data loader. and return generator.
        4. get_data: get batch data set.

    Tensorflow Load Data Tutorial:
        ==> https://www.tensorflow.org/guide/datasets
    """
    def __init__(self, batch_size, sess):
        """
        :param batch_size (int): data batch size.
        :param sess : tensor graph.
        """
        self.batch_size = batch_size
        self.sess = sess

    def add_labels(self, files):
        """
        add image label and change type to ndarray.
        :param files (list): include handle files.
        :return
            features (ndarray): data set.
            labels (ndarray): labels.
        """
        labels = []
        features = []

        for file in files:
            classes = file.split('/')[-1].split('.', 1)[0]
            labels.append([0 if classes == 'dog' else 1])
            features.append([file])

        return np.array(features), np.array(labels)

    def parse_images(self, feature, label):
        """
        parse images.
        1.read file.
        2.decode
        3.resize to (224,224)
        4.pad to (227,227)
        5.normalized to [0-1].

        :param feature (tensor): include image path.
        :param label (tensor): label.
        :return
            image_ones (tensor): parse image result shape is (None,227,227,3)
            label (tensor): parse label result shape is (None,1)
        """
        image_string = tf.read_file(feature[0])
        image_decode = tf.image.decode_jpeg(image_string)
        image_resize = tf.image.resize_images(image_decode, (224, 224))
        # image_normal = tf.image.per_image_standardization(image_resize)
        image_pad = tf.pad(image_resize,([1,0],[1,0],[0,0]))
        image_ones = tf.div(image_pad, 255)

        return image_ones, label

    def get_batch(self, features, labels):
        """
        :param feature (tensor): include image path.
        :param label (tensor): label.

        Note:
            data_set = data_set.repeat():
            ===================================================================
            Infinite get data in sequence.if not set repeat, when the sequence
            is null, gave "End of sequence". means it has traversed the all data
            set.can running the next epoch.
            more information:
            ==>
            https://www.tensorflow.org/guide/datasets#processing_multiple_epochs
            ===================================================================
        """
        features_ = tf.placeholder(features.dtype, features.shape)
        labels_ = tf.placeholder(labels.dtype, labels.shape)

        data_set = tf.data.Dataset.from_tensor_slices((features_, labels_))
        data_set = data_set.map(self.parse_images)
        data_set = data_set.shuffle(1000)
        data_set = data_set.batch(self.batch_size)

        # data_set = data_set.repeat()
        iterator_train = data_set.make_initializable_iterator()
        next_element = iterator_train.get_next()

        # every epoch needs running this code to loading data and shuffled.
        self.sess.run(iterator_train.initializer, feed_dict={features_: features, labels_: labels})

        return next_element

    def get_data(self, files):
        """
        :param feature (tensor): include image path.
        :return next_element (generator):
            it's a generator, running it can get images(batch,227,227,3) and labels(batch,1).
        """
        features, labels = self.add_labels(files)
        next_element = self.get_batch(features, labels)
        return next_element


if __name__ == "__main__":
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    split_data = SplitData(file_dir, Load_samples=100, test_rate=0.3)
    train_files, test_files = split_data()

    with tf.Session() as sess:
        loader = LoadData(50, sess)
        next_element_train = loader.get_data(train_files)
        while 1:
            try:
                images, labels = sess.run(next_element_train)
                print(images[0])
                plt.imshow(images[0])
                plt.show()
                print(labels)
                print(images.shape, labels.shape)

            except tf.errors.OutOfRangeError:
                break
