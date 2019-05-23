import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

"""protocolbuf

Example Message{
    Ffeatures{
        feature{
            key:"name"
            value:{
                bytes_list:{
                    value:"cat"
                }
            }
        }
        feature{
            key:"shape"
            value:{
                int64_list:{
                    value:1266
                    value:1900
                    value:3
                }
            }
        }
        feature{
            key:"data"
            value:{
                bytes_list:{
                    value:0xbe
                    value:0xb2
                    ...
                    value:0x3
                }
            }
        }
    }
}
"""

# 将文件写入TFRecord
def write_test(input,output):
    """
    使用TFRecordWriter 将信息写入 TFRecord文件,
    需要注意的是:
    生成的TFRecord文件往往要比图片原文件要大,因为我们的图片往往都是通过压缩的
    """
    write = tf.python_io.TFRecordWriter(output)
    # 读取图片并解码
    image = tf.read_file(input)
    image = tf.image.decode_jpeg(image) # 除了jped的格式还有png的格式

    with tf.Session() as sess:
        image = sess.run(image)
        shape = image.shape
        print('The Orginal picture shape is:{}'.format(shape))
        # 将图片转换成string,转换成bytes格式也可以,转换后的文件大小相差不大
        image_data = image.tostring()
        
        # 因为下面我们需要将name这个变量存储为beytes_list格式,所以我们这里需要更改为bytes类型
        name = bytes('cat',encoding='utf8')

        # 创建Example对象,并将 Feature 一一对应填充进去.
        example = tf.train.Example(features=tf.train.Features(feature={
            'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape))),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
        }))

        # 将Example实例化成string类型后才可以写入
        write.write(example.SerializeToString())

        write.close()

# 从TFRecord中读取文件

def _parse_record(example_proto):
    features = {
        'name': tf.FixedLenFeature((),tf.string), # FixedLenFeature:获取的是固定长度,如果长度元素为1,则可以填写为()
        'shape': tf.FixedLenFeature([3],tf.int64), # 由于shape list的长度为3,所以这里需要指定固定长度为3.
        'data':tf.FixedLenFeature((),tf.string)
    }

    parsed_features = tf.parse_single_example(example_proto,features=features)

    return parsed_features


def read_test(input_file):

    # 读取tfrecord文件
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(_parse_record) # 使用map函数一一作用与_parse_record
    iterator = dataset.make_one_shot_iterator()

    # 依据官方提示,如果我们迭代的次数过多,我们需要先定义一个先行get_next的方式,否则速度会越来越慢,最终导致资源耗尽
    example = iterator.get_next() 

    with tf.Session() as sess:
        features = sess.run(example)
        name = features['name'].decode('utf8')
        n_h,n_w,n_c = features['shape']
        print('The picture name is:{}'.format(name))
        print('The picture shape is:{}'.format((n_h,n_w,n_c)))

        data = features['data']
        
        # 需要使用numpy将读出来的data更改为数组,才能展示图片,并且依据numpy提示,不能使用np.fromstring,其很不稳定.
        img = np.frombuffer(data,np.uint8)
        img = np.reshape(img,(n_h,n_w,n_c))

        plt.imshow(img);plt.show()

        # 将data从新编码写入本地
        img = tf.image.encode_jpeg(img)
        tf.gfile.GFile('cat_encode.jpg','wb').write(img.eval())


if __name__ == "__main__":
    input_file = 'cat.jpg'
    output_file = 'cat.tfrecord'
    write_test(input=input_file,output=output_file)
    read_test(input_file=output_file)


