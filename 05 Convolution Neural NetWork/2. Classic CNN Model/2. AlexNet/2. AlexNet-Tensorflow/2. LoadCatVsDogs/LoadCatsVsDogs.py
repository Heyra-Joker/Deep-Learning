import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_files(file_dir):
    """
    读取素有的文件,以ndarray的形式储存,以便于下面TF调用,
    data_path 的形状为 (m,1),这里的m指的是样本个数.
    labels 的形状为 (m,)

    Argus:
    ------
        file_dir: 数据路径,需要注意的是文件末尾不要有"/",因为下面"join()"中使用了'/'
    
    Returns:
    -------
        data_path: 包含所有路径的ndarray.
        labels: 包含所有标签,其中cat:0,dog:1.
    """
    data_path = []
    labels = []

    files = os.listdir(file_dir)

    for file in files:
        classes_name,_ = file.split('.',1)
        if classes_name == 'cat':
            labels.append(0)
        else:
            labels.append(1)
        data_path.append(['/'.join((file_dir,file))])
    
    data_path = np.array(data_path)
    labels = np.array(labels)

    return data_path,labels

def _parse_function(filename,labels,n_h=224,n_w=224):
    """
    parse 函数,用于依照map进来的文件路径读取文件,再将其解码为jpeg,最终将图片resize到指定的大小

    Argus:
    ------
        filename: map 进来的图片文件路径
        labels: 对应的图片,放进来不做任何处理,只是为了下面代码dataset.shuffle保证样本与标签一一对应.
        n_h: resize的高
        n_w: resize的宽
    
    Returns:
    -------
        image_resize: 从新resize后的图片,形状为 (batch,n_h,n_w,n_c)
        labels: 传入进来的标签.
    """
    image_string = tf.read_file(filename[0])
    image_decoded = tf.image.decode_jpeg(image_string)
    # resize 图片大小,使用方式 ResizeMethod.BILINEAR,
    image_resize = tf.image.resize_images(image_decoded,(n_h,n_w))

    return image_resize,labels
    

def get_batch(features,labels,n_h,n_w,BATCH_SIZE,epochs):
    """
    使用tf.data.Dataset来读取图片
    更多详细请查看:https://www.tensorflow.org/guide/datasets

    Argus:
    -----
        features: ndarray,其中包含图片路径,shape:(m,1).
        labels: ndarray,其中包含图片标签,cat:0,dog:1.
        n_h: resize 的图片高
        n_w: resize 的图片宽
        BATCH_SIZE: 样本批次
        epochs: 迭代次数

    """
    # 在小数据量的时候可以不设置,
    # 但是大数据量由于可能会达到 tf.GraphDef 协议缓冲区的 2GB 上限,所以官方建议使用替代方案"tf.placeholder".
    features_placeholder = tf.placeholder(features.dtype,features.shape)
    labels_placeholder = tf.placeholder(labels.dtype,labels.shape)
    n_h_placeholder = tf.placeholder(tf.int32,labels.shape)
    n_w_placeholder = tf.placeholder(tf.int32,labels.shape)


    # 装进from_tensor_slices的所有维度必须一致.
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,labels_placeholder,n_h_placeholder,n_w_placeholder))
    dataset = dataset.map(_parse_function) # 将划分的tensor一一执行_parse_function

    # shufle中的参数buffer_size指的是buffer的大小,每次TF取出一条,就会往buffer中增加一条,
    # buffer_size 表示的是buffer存放的item大小.更多详细查看https://juejin.im/post/5b855d016fb9a01a1a27d035
    # 不能设置过大,否则会Shuffle buffer filled.
    dataset = dataset.shuffle(1000) 
    dataset = dataset.batch(BATCH_SIZE) # 每次从buffer中拿出的batch size.
  

    ##########################################################
    # 如果不设置任何值表示无限重复的往buffer中或者队列中加入数据.    
    # 如果设置数值,则当队列重复次数达到该数值就不再重复,             
    # 此时如果再次调用iterator.get_next()就会抛出OutOfRangeError 
    # 一般都设置为无限重复                                     
    ##########################################################
    dataset = dataset.repeat() 

    # 创建迭代器初始化.
    iterator = dataset.make_initializable_iterator()
    # 依照官方建议,使用一个变量缓存iterator.get_next(),否则直接多次调用iterator.get_next(),速度会越来越慢,最终导致资源耗尽.
    next_element = iterator.get_next()
    with tf.Session() as sess:
        n_h = np.array([n_h for i in range(features.shape[0])])
        n_w = np.array([n_w for i in range(features.shape[0])])
        for _ in range(epochs):
            try:
                sess.run(iterator.initializer,feed_dict={features_placeholder:features,labels_placeholder:labels,
                n_h_placeholder:n_h,n_w_placeholder:n_w})
                img,label = sess.run(next_element) # 必须这样写,否则如果再次sess.run,队列就会混乱!!
                plt.imshow(img[0]/255) # 因为我们上面标准化了,所以必须要除上255.才能能显示
                plt.title(label[0])
                plt.show()
            except tf.errors.OutOfRangeError:
                break
            
    
    


if __name__ == "__main__":

    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    data_path,labels = get_files(file_dir)
    get_batch(data_path,labels,n_h=225,n_w=225,BATCH_SIZE=1000,epochs=10)









