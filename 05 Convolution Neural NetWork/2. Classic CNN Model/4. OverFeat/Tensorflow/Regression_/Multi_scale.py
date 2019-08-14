import tensorflow as tf
import numpy as np
from PIL import Image,ImageDraw
from Config import resize_image

class MultiScale_R:
    def __init__(self, sess,test_image, model_dir):
        """
        Arguments:
        ----------
            sess(graph)
            test_image(string): predict image path.
            model_dir(dir): trained model dir.

        """
        self.sess = sess 
        self.model_dir = model_dir
        self.test_image = test_image

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

        # actually, this name need like "W7" or "b7". dont worry about it, it's just my mistake.
        self.W6 = tf.get_variable("W6_1", (6, 6, 1024, 4096))
        self.b6 = tf.get_variable("b6_1", (1, 1, 4096))
        self.W7 = tf.get_variable("W7_1", (1, 1, 4096, 1024))
        self.b7 = tf.get_variable("b7_1", (1, 1, 1024))
        self.W8 = tf.get_variable("W8_1", (1, 1, 1024, 4))
        self.b8 = tf.get_variable("b8_1", (1, 1, 4))

        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_dir)
    def forward(self, data, rate_):
        """
        build R-OverFeat.rate_ is dropout rate.

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

        # OUT
        C8 = tf.nn.conv2d(
            D7, filter=self.W8, strides=(1, 1, 1, 1),
            padding='VALID') + self.b8
        Out = tf.reshape(C8, shape=(-1, 4))
        return Out

    def Mapping_width_height(self,box,h_w,original_w_h):
        """
        Because of our previous equivalent change value,so now we need mapping w and h.
        """

        height,width = h_w
        original_width,original_height = original_w_h
        xmin,ymin,xmax,ymax = box

        n_h = original_height / height
        n_w = original_width / width
        ymin,ymax = ymin * n_h,ymax * n_h
        xmin,xmax = xmin * n_w,xmax * n_w
        box_m = (xmin,ymin,xmax,ymax)

        return box_m
 
    def multi_scale_predict(self):
        """
        Note, In this code I do not used multi-scale and offset pooling.
        """
        h_w = (231,231)
        self.load_W_and_b()
        # load test image 
        iamge = Image.open(self.test_image)
        original_w_h = iamge.size
        iamge = iamge.resize(h_w)
        iamge = np.array(iamge)
        w,h,c = iamge.shape
        iamge = iamge.reshape((1,h,w,c))
        iamge = np.divide(iamge,255)
        # create place holder
        data_ = tf.placeholder(tf.float32,iamge.shape,name='Input')
        rate = tf.placeholder(tf.float32)
        out = self.forward(data_, rate)
        out_ = self.sess.run(out,feed_dict={data_:iamge,rate:0})
        box_ = self.Mapping_width_height(out_[0],h_w,original_w_h)
        self.sess.close()
        return box_


if __name__ == "__main__":
    save_model = '/Users/joker/PycharmProjects/OverFeat_v3/Tensorflow/MODELS/model_R_English_setter/OverFeat_R'
    test_image = '/Users/joker/PycharmProjects/OverFeat_v3/Tensorflow/TEST_IMAGE/English_setter1.jpg'
    with tf.Session() as sess:
        multi_scale = MultiScale_R(sess,test_image,save_model)
        bbox = multi_scale.multi_scale_predict()
        image = Image.open(test_image)
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox,outline='red',width=3)
        image.show()


    
    
