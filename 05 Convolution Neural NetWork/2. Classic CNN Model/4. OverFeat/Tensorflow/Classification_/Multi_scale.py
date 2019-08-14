
import optparse
import numpy as np
from scipy import stats
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from Config import scal_image, n_classes


class ScalImage:
    """
    Implemention scale windows predict.
    """
    def __init__(self, image_path):
        """
        Argument:
        ---------
            image_path(string): need predict image.
        """
        self.image_path = image_path

    def scal_(self):
        """
        scale image to multi-images.

        Return:
        -------
            scal_samples(array): it's have different size of predict image.
        """
        Scal_list = []
        image = Image.open(self.image_path)
        for height, width in scal_image:
            resize_image = image.resize((height, width))
            resize_image = np.array(resize_image)
            w, h, c = resize_image.shape
            resize_image = resize_image.reshape((h, w, c))
            array_image = np.array(resize_image) / 255.
            Scal_list.append(array_image)

        return Scal_list


class MultiScale_C:
    """
    Implementation multi-scale predict.
    """
    def __init__(self,sess,image_path, model_dir):
        """
        Arguments:
        ----------
            image_path(string): predict image path.
            model_dir(dir): trained model dir.

        """
        self.sess = sess
        self.model_dir = model_dir
        self.image_path = image_path
        self.Result = []
        self.flip_classes = {}
        

    def load_classes(self):
        """
        load classes name in file "Clesses_.txt"
        """
        with open('Classes_.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_ = line.strip()
                index, classes = line_.split()
                self.flip_classes[int(index)] = classes

    def load_W_and_b(self):
        """
        Loading weights and bias in trained model.
        """
        self.W1 = tf.get_variable("W1", (11, 11, 3, 96))
        self.b1 = tf.get_variable("b1", (1, 1, 96))
        self.W2 = tf.get_variable("W2", (5, 5, 96, 256))
        self.b2 = tf.get_variable("b2", (1, 1, 256))
        self.W3 = tf.get_variable("W3", (3, 3, 256, 512))
        self.b3 = tf.get_variable("b3", (1, 1, 512))
        self.W4 = tf.get_variable("W4", (3, 3, 512, 1024))
        self.b4 = tf.get_variable("b4", (1, 1, 1024))
        self.W5 = tf.get_variable("W5", (3, 3, 1024, 1024))
        self.b5 = tf.get_variable("b5", (1, 1, 1024))
        self.W6 = tf.get_variable("W6", (6, 6, 1024, 3072))
        self.b6 = tf.get_variable("b6", (1, 1, 3072))
        self.W7 = tf.get_variable("W7", (1, 1, 3072, 4096))
        self.b7 = tf.get_variable("b7", (1, 1, 4096))
        self.W8 = tf.get_variable("W8", (1, 1, 4096, n_classes))
        self.b8 = tf.get_variable("b8", (1, 1, n_classes))
        
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_dir)
        

    def forward(self, data, rate_):
        """
        Because of multi-scale, we have to redefine forward propagation.
        The different shape of "input placeholder".

        Arguments:
        ----------
            data(tensor): testing data.
            rate_(tensor): dropout rate, need set 0 in testing.
        Return:
        -------
            out(tensor): out of forward.
        Note:
        -----
            Because of Multi-scale, the out shape maybe different.
        """
        # Conv1
        C1 = tf.nn.conv2d(
            data, filter=self.W1, strides=(1, 4, 4, 1),
            padding="VALID") + self.b1
        R1 = tf.nn.relu(C1)
        P1 = tf.nn.max_pool(R1,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv2
        C2 = tf.nn.conv2d(
            P1, filter=self.W2, strides=(1, 1, 1, 1),
            padding="VALID") + self.b2
        R2 = tf.nn.relu(C2)
        P2 = tf.nn.max_pool(R2,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv3
        C3 = tf.nn.conv2d(
            P2, filter=self.W3, strides=(1, 1, 1, 1), padding="SAME") + self.b3
        R3 = tf.nn.relu(C3)

        # Conv4
        C4 = tf.nn.conv2d(
            R3, filter=self.W4, strides=(1, 1, 1, 1), padding="SAME") + self.b4
        R5 = tf.nn.relu(C4)

        # Conv5
        C5 = tf.nn.conv2d(
            R5, filter=self.W5, strides=(1, 1, 1, 1), padding="SAME") + self.b5
        R5 = tf.nn.relu(C5)
        P5 = tf.nn.max_pool(R5,
                            ksize=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID")

        # Conv6
        C6 = tf.nn.conv2d(
            P5, filter=self.W6, strides=(1, 1, 1, 1),
            padding="VALID") + self.b6
        R6 = tf.nn.relu(C6)
        D6 = tf.nn.dropout(R6, rate=rate_)

        # Conv7
        C7 = tf.nn.conv2d(
            D6, filter=self.W7, strides=(1, 1, 1, 1),
            padding="VALID") + self.b7
        R7 = tf.nn.relu(C7)
        D7 = tf.nn.dropout(R7, rate=rate_)

        # Conv8
        C8 = tf.nn.conv2d(
            D7, filter=self.W8, strides=(1, 1, 1, 1),
            padding="VALID") + self.b8
        out = tf.reshape(C8, (-1, n_classes))
        return out

    def offset(self, image):
        """
        implementation offset pooling.
        reference:
            Paragraph 3.3 (Multi-scale classification) Figure 3
            "https://arxiv.org/pdf/1312.6229.pdf"
        """
        for x in range(0, 3):
            for y in range(0, 3):
                offset_image = image[x:, y:, :]
                # input network...
                yield offset_image

    def predict(self, data):
        """
        Start predicting specify data.

        Argument:
        ---------
            data(array): need predict's image.
        """
        h, w, c = data.shape
        data = data.reshape((1, h, w, c))
        X = tf.placeholder(tf.float32, data.shape)
        rate = tf.placeholder(tf.float32)
        out = self.forward(X, rate)

        softmax_out = tf.nn.softmax(out)
        predict_top_k = tf.math.top_k(softmax_out, k=1)
        _, indices = self.sess.run(predict_top_k, feed_dict={X: data, rate: 0})

        # Gets the number of public (Mode)
        counts = stats.mode(indices)[0][0][0]
        self.Result.append(counts)

    def multi_scale_predict(self):
        """
        Runing...
        """
        self.load_classes()
        # loade tensorflow meta.
        self.load_W_and_b()
        # scal image.
        _scal_images = ScalImage(self.image_path)
        scal_samples = _scal_images.scal_()
        # offset
        for sample in scal_samples:
            offset_ = self.offset(sample)
            while True:
                try:
                    offset_image = next(offset_)
                    self.predict(offset_image)
                except:
                    break
        predict_label = stats.mode(self.Result)[0][0]
        classes_name = self.flip_classes[predict_label]
        return classes_name

if __name__ == "__main__":
    for i in range(1,5):
        test_image_path = '/Users/joker/PycharmProjects/OverFeat_v3/Tensorflow/TEST_IMAGE/English_setter%d.jpg'%i
        model_path_C = '/Users/joker/PycharmProjects/OverFeat_v3/Tensorflow/MODELS/model_C/OverFeat'
        tf.reset_default_graph()
        with tf.Session() as sess:
            multi_C = MultiScale_C(sess,test_image_path, model_path_C)
            classes_name = multi_C.multi_scale_predict()
            image = Image.open(test_image_path)
            fnt = ImageFont.truetype('../LatienneSwaT.ttf', 30)
            # get a drawing context
            draw = ImageDraw.Draw(image)
            # draw text, half opacity
            true_name = test_image_path.split('/')[-1].split('.')[0]
            text = 'Predict:%s\nTure:%s' % (classes_name, true_name)
            draw.text((0, 0), text, font=fnt, fill='red')
            image.show()
        
        