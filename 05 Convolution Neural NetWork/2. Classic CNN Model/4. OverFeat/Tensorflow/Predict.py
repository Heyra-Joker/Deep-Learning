from Classification_.Multi_scale import MultiScale_C
from Regression_.Multi_scale import  MultiScale_R

import optparse
import tensorflow as tf
from PIL import Image,ImageDraw,ImageFont

def show_predict(image_path,name,bbox,is_save=False):
    """
    show predict result.
    Arguments:
    ----------
        name(str): model predict name.
    """
    image = Image.open(image_path)
    fnt = ImageFont.truetype('../LatienneSwaT.ttf', 30)
    # get a drawing context
    draw = ImageDraw.Draw(image)

    # draw text, half opacity
    true_name = image_path.split('/')[-1].split('.')[0]
    text = 'Predict:%s\nTure:%s' % (name, true_name)
    draw.text((0, 0), text, font=fnt, fill='red')

    # rectange
    draw.rectangle(bbox,outline='blue',width=2)
    image.show()

    if is_save:
        # save result image
        save_path_ = 'RESULT/res_{}.jpg'.format(true_name)
        image.save(save_path_)
        print('predict result saved path is "{}" '.format(save_path_))

def predict():
    parser = optparse.OptionParser("""-f <test image file path>\n[-s] <is save predict result>""")
    parser.add_option('-f',
                      dest='test_image_path',
                      type='string',
                      help='Specify test image file path')
    
    parser.add_option('-s',
                      dest='is_save_res',
                      type='string',
                      help='Is saved peridect result')

    (options, _) = parser.parse_args()

    test_image_path = options.test_image_path
    is_save_res = options.is_save_res
    if is_save_res == 'True':
        is_save = True

    if test_image_path is None:
        print(parser.usage)
        exit(0)
    else:
        model_path_C = 'MODELS/model_C/OverFeat'
        model_path_R = 'MODELS/model_R_English_setter/OverFeat_R'
        with tf.Session() as sess:
            multi_C = MultiScale_C(sess,test_image_path, model_path_C)
            classes_name = multi_C.multi_scale_predict()

        tf.reset_default_graph()
        with tf.Session() as sess:
            multi_R = MultiScale_R(sess,test_image_path,model_path_R)
            bbox = multi_R.multi_scale_predict()
        show_predict(test_image_path,classes_name,bbox,is_save)


if __name__ == '__main__':
    predict()
    














