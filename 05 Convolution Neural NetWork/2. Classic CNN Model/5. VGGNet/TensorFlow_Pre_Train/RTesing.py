import warnings
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw

from keras.applications import VGG16



class SingleScaleTesting:
    def __init__(self, model_dir, test_file_path, image_scale=256):
        self.sess = tf.Session()
        self.model_dir = model_dir
        self.test_file_path = test_file_path
        self.image_scale = image_scale
        self.n_classes = 4
        self.vgg_model = VGG16(include_top=False, weights='imagenet')

    def PlotImage(self, bboxs):
        image = Image.open(self.test_file_path)
        # image = image.resize((300,300))
        w,h = image.size
        draw = ImageDraw.Draw(image)
        for bbox in bboxs[0]:
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = xmin * w, ymin * h , xmax * w, ymax * h
            print(xmin, ymin, xmax, ymax)
            draw.rectangle([xmin, ymin, xmax, ymax],outline='red',width=2)
        image.show()

    def ReadFile(self, img, scale):
        """
        Used Tensorflow to read image file.
        Arguments:
        ---------
           img(str): target image.
           scale(int): resize scale.
        
        Return:
        ------
            img: decode image and reshape to scale
        """
        img = tf.convert_to_tensor(img)
        img = tf.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (scale, scale))
        img = tf.expand_dims(img,axis=0)
        img = tf.math.divide(img, 255)
        return img

    def keras_per_model(self, data):
        data = self.sess.run(data)
        features = self.vgg_model.predict(data)
        return features

    def Reshape_weights(self, FC_Prams):
        """
        Since we are using the fully connected layer during the training phase, 
        we need to redefine the shape here.

        Argument:
        --------
            FC_Prams(tensor): load weights in model file.
        Return:
        ------
            ReshapeFCPrams(tensor): redefine the shape weights and bais. 
        """
        (W14, b14, W15, b15, W16, b16) = FC_Prams
        W14 = tf.reshape(W14, (7, 7, 512, 4096))
        b14 = tf.reshape(b14, (1, 1, 4096))
        W15 = tf.reshape(W15, (1, 1, 4096, 4096))
        b15 = tf.reshape(b15, (1, 1, 4096))
        W16 = tf.reshape(W16, (1, 1, 4096, self.n_classes))
        b16 = tf.reshape(b16, (1, 1, self.n_classes))
        ReshapeFCPrams = (W14, b14, W15, b15, W16, b16)
        return ReshapeFCPrams

    def Load_weights(self):
        """
        Load weights and bias.
        """
        W14 = tf.get_variable('W14', (7 * 7 * 512, 4096))
        b14 = tf.get_variable('b14', (1, 4096))
        W15 = tf.get_variable('W15', (4096, 4096))
        b15 = tf.get_variable('b15', (1, 4096))
        W16 = tf.get_variable('W16', (4096, self.n_classes))
        b16 = tf.get_variable('b16', (1, self.n_classes))

        saver = tf.train.Saver([W14, b14, W15, b15, W16, b16])
        # restore target weights and bias.
        saver.restore(self.sess, self.model_dir)

        FC_Prams = (W14, b14, W15, b15, W16, b16)

        ReshapeFCPrams = self.Reshape_weights(FC_Prams)
        return ReshapeFCPrams
    
    def CONV(self, data, W, b):
        C = tf.add(tf.nn.conv2d(data, W, (1, 1, 1, 1), "VALID"), b)
        return C

    def Net(self, data, parameters):
        """
        Build Net.
        """
        (W14, b14, W15, b15, W16, b16) = parameters
        # FCN14
        C14 = self.CONV(data, W14, b14)
        R14 = tf.nn.relu(C14)
        # FCN15
        C15 = self.CONV(R14, W15, b15)
        R15 = tf.nn.relu(C15)
        # FCN16
        C16 = self.CONV(R15, W16, b16)
        out = tf.reshape(C16, (1, -1, self.n_classes))
        return out

    def Predict(self):
        """
        绝对不能再加载权重完毕之后再一次初始化
        """
        print("WARNING: Just Specify Your Trained Regression Classes Name.")

        # load parameters
        params = self.Load_weights()
        image = self.ReadFile(self.test_file_path, self.image_scale)
        
        pre_image = self.keras_per_model(image)
        Out = self.Net(pre_image, params)
        predict = np.exp(self.sess.run(Out))
        self.PlotImage(predict)

if __name__ == "__main__":
    model_dir = 'ModelRVgg16/R_VGG16.ckpt'
    test_file_path = '../TEST_IMAGES/n02116738_678.jpg'
    image_scale = 224
    single_scale_testing = SingleScaleTesting(model_dir, test_file_path, image_scale=image_scale)
    single_scale_testing.Predict()
        

    
    
