import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def Predict(model_dir, picture_path):
    """
    Implemention Predict Image.

    Argus:
    -----
        model_dir: Tensorflow saver model dir path. it's like 'model_/alexnet'.
        picture_path: precit picture path.
    """

    Predict_image = np.zeros((10, 224, 224, 3))

    image = Image.open(picture_path)
    resiz_image = image.resize((256, 256))

    # left up
    left_up = resiz_image.crop((0, 0, 224, 224))  # (left,up,right,bottom).
    left_up_transpose = left_up.transpose(Image.FLIP_LEFT_RIGHT)
    Predict_image[0, ...] += np.array(left_up)
    Predict_image[1, ...] += np.array(left_up_transpose)
    # right up
    right_up = resiz_image.crop((256 - 224, 0, 256, 224))
    right_up_transpose = right_up.transpose(Image.FLIP_LEFT_RIGHT)
    Predict_image[2, ...] += np.array(right_up)
    Predict_image[3, ...] += np.array(right_up_transpose)
    # left bottom
    left_bottom = resiz_image.crop((0, 256 - 224, 224, 256))
    left_bottom_transpose = left_bottom.transpose(Image.FLIP_LEFT_RIGHT)
    Predict_image[4, ...] += np.array(left_bottom)
    Predict_image[5, ...] += np.array(left_bottom_transpose)
    # right bottom
    right_bottom = resiz_image.crop((256 - 224, 256 - 224, 256, 256))
    right_bottom_transpose = right_bottom.transpose(Image.FLIP_LEFT_RIGHT)
    Predict_image[6, ...] += np.array(right_bottom)
    Predict_image[7, ...] += np.array(right_bottom_transpose)
    # center
    pixe_ = 256 - 224
    center = resiz_image.crop((pixe_ / 2, pixe_ / 2, 256 - pixe_ / 2, 256 - pixe_ / 2))
    center_transpose = center.transpose(Image.FLIP_LEFT_RIGHT)
    Predict_image[8, ...] += np.array(center)
    Predict_image[9, ...] += np.array(center_transpose)


    # Get Save Model
    tf.reset_default_graph()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_dir + '.meta')
        new_saver.restore(sess, model_dir)
        # tf.get_collection() return a list.
        out = tf.get_collection('pre_network')[0]

        graph = tf.get_default_graph()

        # get Variable X by tensor name.
        X = graph.get_operation_by_name('Input').outputs[0]
        # get Variable rate by tensor name.
        # Notice, we do not load Variable y !.
        rate = graph.get_operation_by_name('rate').outputs[0]

        # pad data to shape (227,227)
        data = np.pad(Predict_image, pad_width=((0, 0), (1, 2), (1, 2), (0, 0)), mode='constant')
        data = data / 255
        plt.imshow(data[5])
        plt.show()
        # predict
        OUT = sess.run(out, feed_dict={X: data, rate: 0})

        sigmoid_out = 1. / (1 + np.exp(-OUT))
        roud_ = np.round(sigmoid_out)
        classes_cat_rate = np.sum(roud_) / 10
        classes_dog_rate = 1 - classes_cat_rate

        print('The picture is cat rate: {:.4f} %'.format(classes_cat_rate * 100))
        print('The picture is dog rate: {:.4f} % '.format(classes_dog_rate * 100))

if __name__ == "__main__":
    model_dir = 'model/alexNet'
    picture_path = 'MASK/dog_face.jpg'
    Predict(model_dir, picture_path)

