import random
import numpy as np
import tensorflow as tf

from Load_Flies import LoadFiles_
from Classes import classes_name

"""
Build Data Loader Generator.
It's have two routes:
    1. regression routes: get_batch ==> _zoom ==> _augmented_bbox(train) ==> _parse.
    2. classification routes: get_batch ==> _zoom ==> _augmented_label(train) ==> _parse ==> _one_hot.

Note:
    1. In Classification, The labels need changed hot.
    2. In Rregression, The image will resize to (224,224) not (256,256) and crop to (224,224).
"""
class Loader:
    
    def __init__(self,sess,batch_size):
        self.sess = sess
        self.batch_size = batch_size
    
    def _parse(self,image,target):
        image = tf.math.divide(image,255)
        return image,target
    
    def _one_hot(self,image,target):
        target = tf.squeeze(target)
        target = tf.cast(target,tf.uint8)
        target = tf.one_hot(target,len(classes_name))
        return image,target

    def _zoom(self,image,target):
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.resize(image,(self.S,self.S))
        image = tf.image.random_crop(image,(224,224,3))
        return image,target
    
    def _augmented_bbox(self,image,target):
        """
        do not use flip left right !
        """
        image = tf.image.random_hue(image,max_delta=0.1)
        return image,target

    def _augmented_label(self,image,target):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image,max_delta=0.1)
        return image,target

    def get_batch(self,data,targets,data_mode='train',bbox=False):
        # create placeholder
        _data = tf.placeholder(data.dtype,data.shape)
        _targets = tf.placeholder(targets.dtype,targets.shape)
        # create dataset.
        dataset = tf.data.Dataset.from_tensor_slices((_data,_targets))

        # handle data.
        # want get bboxs
        if bbox:
            self.S = 224
            dataset = dataset.map(self._zoom)
            if data_mode == 'train':
                dataset = dataset.map(self._augmented_bbox)
            dataset = dataset.map(self._parse)

        else:
            self.S = random.randint(256,512)
            # want get label
            dataset = dataset.map(self._zoom)
            if data_mode == 'train':
                dataset = dataset.map(self._augmented_label)
            dataset = dataset.map(self._parse)
            dataset = dataset.map(self._one_hot)


        # shuffle and batch.
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        self.sess.run(iterator.initializer, feed_dict={_data:data,_targets:targets})

        return next_element


if __name__ == "__main__":
    from PIL import Image,ImageDraw
    Annotation_dir = '/Users/joker/jokers/DataSet/stanford-dogs-dataset/Annotation'
    test_file_save_path = 'TEST_FILES'
    sess = tf.Session()
    
    # bbox
    loadfiles_ = LoadFiles_(Annotation_dir,(0.7,0.2,0.1),
    test_file_save_path,target_mode='bboxs',bbox_classes='n02116738-African_hunting_dog')

    samples_train,samples_val = loadfiles_.load_files()
    loader = Loader(sess,64)
    # for epoch ...
    next_element = loader.get_batch(samples_train[0],samples_train[1],data_mode='val',bbox=True)
    while 1:
        try:
            imgs,labels = sess.run(next_element)
            img = Image.fromarray((imgs[0]*255).astype('uint8'))
            draw = ImageDraw.Draw(img)
            xmin,ymin,xmax,ymax = labels[0] * 224
            draw.rectangle([xmin,ymin,xmax,ymax],outline='red',width=3)
            img.show()
        except tf.errors.OutOfRangeError:
            break
    
    """
    # labels
    loadfiles_ = LoadFiles_(Annotation_dir,(0.7,0.2,0.1),
    test_file_save_path,target_mode='labels')

    samples_train,samples_val = loadfiles_.load_files()
    loader = Loader(sess,64)
    # for epoch ...
    next_element = loader.get_batch(samples_val[0],samples_val[1],data_mode='train',bbox=False)
    while 1:
        try:
            imgs,labels = sess.run(next_element)
            img = Image.fromarray((imgs[0]*255).astype('uint8'))
            print(labels.shape)
            print(np.argmax(labels[0]))
            img.show()
            break
        except tf.errors.OutOfRangeError:
            break
    """











    

