import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

class Load_data:
    def __init__(self,file_dir,n_h,n_w,BATCH_SIZE,sess):
        self.file_dir = file_dir
        self.n_h = n_h
        self.n_w = n_w
        self.BATCH_SIZE = BATCH_SIZE
        self.total_sample = 0
        self.cat_sample = 0
        self.dog_sample = 0
        self.sess = sess
        features,labels = self.get_files()
        self.next_element = self.get_batch(features,labels)
        
    def get_files(self):
        """
        Get files fo given file_dir parameter.

        Returns:
        -------
            data_path: Include picture file path  and it's a  ndarray,the shape is  (m_sample,1).
            labels: incloud labels,cat:0,dog:1.the shape is (m_sample,)
        Note:
        ----
            Given file_dir parameter have not "/" in the end.like "../../train"
        """
        data_path = []
        labels = []

        files = os.listdir(file_dir)
        self.total_sample = len(files)
        for file in files:
            classes_name,_ = file.split('.',1)
            if classes_name == 'cat':
                labels.append(0)
                self.cat_sample += 1
            else:
                labels.append(1)
                self.dog_sample += 1
            data_path.append(['/'.join((file_dir,file))])

        data_path = np.array(data_path)
        labels = np.array(labels)

        return data_path,labels
    
    def _parse_function(self,filename,labels,n_h=224,n_w=224):
        """
        parse fuction.

        Argus:
        ------
            filename: data_path,the shape is (batch,1), incloud picture file path.
            labels: picture labels, not do anything !Make sure the labels one by one training he picture in the shuffle data.
            n_h: resize's height.
            n_w: resize' width.

        Returns:
        -------
            image_resize: result of image riszed ,the shape is (batch,n_h,n_w,n_c).
            labels: labels.
        """
        image_string = tf.read_file(filename[0])
        image_decoded = tf.image.decode_jpeg(image_string)
        # resize,using ResizeMethod.BILINEAR.
        image_resize = tf.image.resize_images(image_decoded,(n_h,n_w))
        return image_resize,labels
    
    
    def get_batch(self,features,labels):
        """
        Using tf.data.Dataset to read file.
        More Information:https://www.tensorflow.org/guide/datasets

        Argus:
        -----
            features: ndarray,incloud picture path ,shape:(m,1).
            labels: ndarray,incloud picture labes.cat:0,dog:1.
            n_h: resize's heights.
            n_w: resize's widths.
            BATCH_SIZE: mini batch size.
        """
        features_placeholder = tf.placeholder(features.dtype,features.shape)
        labels_placeholder = tf.placeholder(labels.dtype,labels.shape)
        n_h_placeholder = tf.placeholder(tf.int32,labels.shape)
        n_w_placeholder = tf.placeholder(tf.int32,labels.shape)


        # every parameter's shape must be same!.
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,labels_placeholder,n_h_placeholder,n_w_placeholder))
        dataset = dataset.map(self._parse_function) 
        dataset = dataset.shuffle(1000) 
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.repeat() 

        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        
        n_h = np.array([self.n_h for i in range(features.shape[0])])
        n_w = np.array([self.n_w for i in range(features.shape[0])])
        self.sess.run(iterator.initializer,feed_dict={features_placeholder:features,labels_placeholder:labels,
            n_h_placeholder:n_h,n_w_placeholder:n_w})
        
        return next_element
            
    def Get_next_(self):
        
        try:
            img,label = self.sess.run(self.next_element)
            return img,label
        
        except tf.errors.OutOfRangeError:
            
            print('The Queue is empy !')


def init_parameters():
    
    init_w = tf.keras.initializers.glorot_normal(seed=1)
    init_b = tf.keras.initializers.zeros()
    W1 = tf.get_variable('W1',[11,11,3,96],initializer=init_w)
    b1 = tf.get_variable('b1',[1,1,96],initializer=init_w)
    W2 = tf.get_variable('W2',[5,5,96,256],initializer=init_w)
    b2 = tf.get_variable('b2',[1,1,256],initializer=init_b)
    W3 = tf.get_variable('W3',[3,3,256,384],initializer=init_w)
    b3 = tf.get_variable('b3',[1,1,384],initializer=init_b)
    W4 = tf.get_variable('W4',[3,3,384,384],initializer=init_w)
    b4 = tf.get_variable('b4',[1,1,384],initializer=init_b)
    W5 = tf.get_variable('W5',[3,3,384,256],initializer=init_w)
    b5 = tf.get_variable('b5',[1,1,256],initializer=init_b)
    
    parameters = W1,b1,W2,b2,W3,b3,W4,b4,W5,b5
    
    return parameters

def forward(data,parameters,rate):
    
    W1,b1,W2,b2,W3,b3,W4,b4,W5,b5 = parameters

    C1 = tf.nn.conv2d(data,W1,strides=[1,4,4,1],padding="VALID",name='CONV1')+b1
    R1 = tf.nn.relu(C1)
    P1 = tf.nn.max_pool(R1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name='POOL1')
    LRN1 = tf.nn.lrn(P1, depth_radius=5, bias=2, alpha=1e-4, beta=0.75,name='LRN1')
    
    C2 = tf.nn.conv2d(LRN1,W2,strides=[1,1,1,1],padding='SAME',name="CONV2")+b2
    R2 = tf.nn.relu(C2)
    P2 = tf.nn.max_pool(R2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name='POOL2')
    LRN2 = tf.nn.lrn(P2, depth_radius=5, bias=2, alpha=1e-4, beta=0.75,name="LRN2")
    
    C3 = tf.nn.conv2d(LRN2,W3,strides=[1,1,1,1],padding="SAME",name="CONV3") + b3
    R3 = tf.nn.relu(C3)
    
    C4 = tf.nn.conv2d(R3,W4,strides=[1,1,1,1],padding="SAME",name="CONV4") + b4
    R4 = tf.nn.relu(C4)
    
    C5 = tf.nn.conv2d(R4,W5,strides=[1,1,1,1],padding="SAME",name="CONV5") + b5
    R5 = tf.nn.relu(C5)
    P5 = tf.nn.max_pool(R5,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="POOL5")
    
    Flatten = tf.layers.flatten(P5,name='Flatten')
    
    F6 = tf.contrib.layers.fully_connected(Flatten,num_outputs=4096)
    R6 = tf.nn.relu(F6)
    D6 = tf.nn.dropout(R6,rate=rate,name='DROP6')
    
    F7 = tf.contrib.layers.fully_connected(D6,num_outputs=4096)
    R7 = tf.nn.relu(F7)
    D7 = tf.nn.dropout(R7,rate=rate,name='DROP7')
    
    F8 = tf.contrib.layers.fully_connected(D7,num_outputs=1000)
    R8 = tf.nn.relu(F8)
    D8 = tf.nn.dropout(R8,rate=rate,name='DROP8')
    
    out = tf.contrib.layers.fully_connected(D8,num_outputs=1,activation_fn=None)
    
    return out

def Model(epochs,lr,RATE,BATCH_SIZE,file_dir,resize_shape):
    
    ops.reset_default_graph()
    n_h,n_w = resize_shape
    X = tf.placeholder(tf.float32,[None,227,227,3],name='Input')
    y = tf.placeholder(tf.float32,[None,1],name="Labels")
    rate = tf.placeholder(tf.float32,name='rate')
    
    parameters  = init_parameters()
    stand_X = tf.image.per_image_standardization(X)
    out = forward(stand_X,parameters=parameters,rate=rate)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)
    
    sigmoid_out = tf.nn.sigmoid(out)
    predict = tf.round(sigmoid_out)
    correct = tf.equal(predict,y)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
    saver = tf.train.Saver()
    tf.add_to_collection('pre_network',out)
    init = tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        sess.run(init)
        load_ = Load_data(file_dir=file_dir,n_h=n_h,n_w=n_w,BATCH_SIZE=BATCH_SIZE,sess=sess)
        total_sample = load_.total_sample
        N = total_sample// BATCH_SIZE
        for epoch in range(epochs):
            
            for i in range(N):
                print('Batch Size:{} [{}/{}/{}-{}]\r'.format(BATCH_SIZE,epochs,epoch,N,i),end="",flush=True)
                mini_x,mini_y = load_.Get_next_()
                mini_x = np.pad(mini_x,pad_width=((0,0),(1,2),(1,2),(0,0)),mode='constant')
                
                mini_y = mini_y.reshape(-1,1)
                sess.run([optimizer],feed_dict={X:mini_x,y:mini_y,rate:RATE})
                    
                data,labels = load_.Get_next_()
                data = np.pad(data,pad_width=((0,0),(1,2),(1,2),(0,0)),mode='constant')
                labels = labels.reshape(-1,1)
                acc_train,loss_train = sess.run([accuracy,cost],feed_dict={X:data,y:labels,rate:0})
                
            print('[{}/{}] loss train:{:.4f} acc train:{:.4f}'.format(epoch+1,epochs,loss_train,acc_train))
        saver.save(sess,'model/alexnet')


if __name__ == "__main__":
    file_dir = 'train'
    Model(epochs=13,lr=1e-4,RATE=0.3,BATCH_SIZE=100,file_dir=file_dir,resize_shape=(224,224))