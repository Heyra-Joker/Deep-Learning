import os
import numpy as np
from scipy import stats
import tensorflow as tf
from keras.applications import VGG16
from PIL import Image, ImageFont, ImageDraw


from Classes import flip_classes_name

"""
Classification of Testing.

In the Multi-Tesing have two routes:
    1. Evaluate:
    ===========================================================================================
    We use the Top-K criterion for each scale, 
    and finally we obtain the highest frequency Top-K as the predicted value for many scales.

    Note:
    Since we use an iterative calculation of one image, 
    the speed will be very slow during the test phase.
    ===========================================================================================

    2. Predict:
    ===========================================================================================
    In the prediction mode, we first use TensorFlow to read the image, 
    then we still use Top-k under each scale to get top-k value, 
    and finally get top-k as the predicted value at all scales.
    ===========================================================================================

    3.Note:
    ========================
    DO NOT USING "preprocess_input(img)"
"""

class MultiTesting:
    def __init__(self, batch_size, model_dir, evaluate=True, test_file_path=None, K=5):
        """
        Arguments:
        ----------
            batch_size(int): batch size.
            model_dir(str): VGG-16 model weights file.
            evaluate(bool): choose predict or evaluate if ture.
            test_file_path(str): test file path, if mode is evluate, it is dir path else picture file path.
            K(int): predict top-k.
        """
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.evaluate = evaluate
        self.test_file_path = test_file_path
        self.n_classes = 120
        self.scale_tuple = (224, 256)
        self.K = K

        self.GLOBAL_ACCURACY = 0
        self.vgg_model = VGG16(include_top=False, weights='imagenet')
    
    def PlotImage(self, sort):
        """
        Used PIL to plot Image.
        Argument:
        --------
            sort(array): highest frequency Top-K in all scale result.
        """
        image = Image.open(self.test_file_path)
        image = image.resize((500, 500))
        fnt = ImageFont.truetype('../NotoSerifCJKSCBlack.ttf', 15)
        # get a drawing context
        draw = ImageDraw.Draw(image)
        text = 'Predict Top-%d'% self.K
        draw.text((0, 0), text, font=fnt, fill='red')
        for index, label in enumerate(sort):
            classes_name = flip_classes_name[label]
            # draw text, half opacity
            text = '%d-Predict:%s' % (index +1,classes_name)
            print(text)
            draw.text((0, (index+1)*20), text, font=fnt, fill='#039BE5')
        image.show()

    
    def ReadFile(self, img, scale, divided=False):
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
        if divided:
            img = tf.math.divide(img, 255)
        img = tf.image.resize(img, (scale, scale))
        img = tf.reshape(img, (1, scale, scale, 3))
        return img

    def keras_per_model(self, data):
        
        features = self.vgg_model.predict(data)
        return features
    
    def Load_test_file(self):
        """
        Load test files and return image data, labels.
        """
        image_path = os.path.join(self.test_file_path,
                                  'labels_test_images.npy')
        labels_path = os.path.join(self.test_file_path,
                                   'labels_test_labels.npy')
        image = np.load(image_path)
        labels = np.load(labels_path)
        self.N = image.shape[0]
        return image, labels

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
        return C16
    
    def STop_k(self, image):
        """
        Get Current Scale top-k.
        """
        image = tf.reshape(image, (1, -1, self.n_classes))
        softmax = tf.nn.softmax(image, -1)
        values, indexs = tf.math.top_k(softmax, self.K)
        values, indexs  = tf.reshape(values, (1,-1)), tf.reshape(indexs, (1,-1))
        return values, indexs
    
    def Top_k(self,hstack_indexs):
        """
        Get top-k in all scale.
        """
        # get top-k label
        itemfreq = stats.itemfreq(np.squeeze(hstack_indexs))
        top_k = itemfreq[np.argsort(-itemfreq[:,1])][:self.K][:,0]
        return top_k
    
    def ScalePredict(self, image):
        """
        Predict in current scale.
        """
        image = self.sess.run(image)
        pre_image = self.keras_per_model(image)
        out = self.Net(pre_image, self.Parms)
        values, indexs = self.STop_k(out)
        return values, indexs
    
    def Evaluate(self):
        """
        Model pf Evaluate.
        """
        images, labels = self.Load_test_file()
        N = images.shape[0]
        for index,img in enumerate(images):
            print('[*] Testing {}-{}\r'.format(index + 1, N),end="", flush=True)
            hstack, is_first = None, True
            label = labels[index]
            for scale in self.scale_tuple:
                image = self.ReadFile(img, scale)
                values, indexs = self.ScalePredict(image)
                indexs = self.sess.run(indexs)
                del values
                if is_first:
                    hstack,is_first = indexs, False
                else:
                    hstack = np.hstack((hstack, indexs))
            # get top-k
            top_k = self.Top_k(hstack)
            
            if label in top_k:
                self.GLOBAL_ACCURACY += 1
        print()
        print('[+] The Test Accuracy is:%f'%(self.GLOBAL_ACCURACY / N))
    
    def Predict(self):
        """
        Prediction of model.
        """
        hstack_indexs, is_first = None, True
        for scale in self.scale_tuple:
            image = self.ReadFile(self.test_file_path, scale, divided=True)
            _, indexs = self.ScalePredict(image)
            indexs = self.sess.run(indexs)
            if is_first:
                hstack_indexs, is_first = indexs, False
            else:
                hstack_indexs = np.hstack((hstack_indexs, indexs))
        top_k = self.Top_k(hstack_indexs)
        self.PlotImage(top_k)
        
    def Testing(self):
        self.Parms = self.Load_weights()
        if self.evaluate:
            # running evaluate mode.
            self.Evaluate()
        else:
            # running predict mode.
            self.scale_tuple = (256,284,512)
            self.Predict()


if __name__ == "__main__":
    """
    # test data set.
    test_file_path = 'TEST_FILES'
    model_dir = 'ModelCVgg16/C_VGG16.ckpt'
    multi_testing = MultiTesting(batch_size=10, model_dir=model_dir, evaluate=True, test_file_path=test_file_path, K=5)
    multi_testing.Testing()
    """
      
    # test predict
    test_file_path = '../TEST_IMAGES/n02113023_15462.jpg'
    model_dir = 'ModelCVgg16/C_VGG16.ckpt'
    multi_testing = MultiTesting(batch_size=1, model_dir=model_dir, evaluate=False, test_file_path=test_file_path, K=5)
    multi_testing.Testing()
    
