import numpy as np
from PIL import Image, ImageDraw

o_width = 500
o_height = 375
r_width = 448
r_height = 448
# positions = ((54,25,454,315), (318,37,489,161),(369,1,458,130))
positions = ((1,19,460,348), (258,175,334,336))
# image = Image.open("/Users/joker/jokers/DataSet/VOCdevkit_train/VOC2012/JPEGImages/2007_000364.jpg")
image = Image.open("/Users/joker/jokers/DataSet/VOCdevkit_train/VOC2012/JPEGImages/2007_002565.jpg")
image = image.resize((r_width,r_height))
label = np.zeros((7,7,25))
draw = ImageDraw.Draw(image)
for xmin,ymin,xmax,ymax in positions:
    # 等比例缩放
    w,h = r_width / o_width, r_height / o_height
    xmin,ymin,xmax,ymax = xmin * w, ymin * h, xmax * w, ymax * h
    draw.rectangle((xmin, ymin, xmax, ymax), outline='blue',width=3)
    center_x = (xmax + xmin) /2
    center_y = (ymax + ymin)/2
    # 绘制xcenter,ycenter
    draw.point((center_x,center_y),fill='red')
    # 获取一个cell含有x,y这个点
    x_id = int(center_x // 64)
    y_id = int(center_y // 64)
    print(x_id,y_id)
    # 得到x和y
    x = (center_x - 64 * x_id) / 64
    y = (center_y - 64 * y_id) / 64
    print(x,y)
    # 得到 w,h
    _w = (xmax - xmin) / 488
    _h = (ymax - ymin) / 488
    print(_w, _h)
    # 获取到包含对象的cell的index,如果超出7个则截断,-1是因为某个cell只包含一小部分object
    incloud_object_xmin_id = np.maximum((x_id - (((xmax - xmin) / 2) // 64)),0)
    incloud_object_xmax_id = np.minimum((x_id + (((xmax - xmin) / 2) // 64)),6)
    incloud_object_ymin_id = np.maximum((y_id - (((ymax - ymin) / 2) // 64)),0)
    incloud_object_ymax_id = np.minimum((y_id + (((ymax - ymin) / 2) // 64)),6)
    print(incloud_object_xmin_id, incloud_object_xmax_id, incloud_object_ymin_id, incloud_object_ymax_id)
    # 整合bboexs
    bboxes = [x,y,_w,_h]
    # 模拟当前object得到的标签index
    classes_id = np.random.randint(0,20)
    print(classes_id)
    # 整合label,label排列符合:[1-confidences, 4-bboexs,20-classes]
    # 由于一个cell内可能同时含有多个xy,而confidence只需要设置一次即可
    if label[x_id,y_id,0] == 1:
        continue
    else:
        # 由于这张测试图片中,两个类:人和摩托都在一个cell内,所以最终的label只有一个类,这是yolov1的缺陷.
        label[x_id,y_id,0] = 1
        label[x_id,y_id,1:5] = bboxes
        label[x_id,y_id,5 + classes_id] = 1

    print('**********')
    
# 绘制grid cell
for i in range(6):
    x1 = (i + 1) * 64
    y1 = 0
    x2 = x1
    y2 = 448
    draw.line((x1,y1,x2,y2),width=2,fill='red')
    draw.line((y1,x1,y2,x2),width=2,fill='red')

image.show()
print(label)
image = Image.open("/Users/joker/jokers/DataSet/VOCdevkit_train/VOC2012/JPEGImages/2007_000364.jpg")
image = image.resize((r_width,r_height))
draw = ImageDraw.Draw(image)

# 依照label在原图上重现centerx,centery,bounding box. classes,也就是说假设这里的label就是预测出来的值.
label = label.reshape((7,7,25))
res = np.nonzero(label)
N = res[0].shape[0] // 6
for i in range(N):
    start = i * 6
    end = (i + 1) * 6
    # 获取x_id,y_id
    x_id,y_id = res[0][start:end][0], res[1][start:end][0]
    # 获取类别索引,记得-5,因为在放置类别索引的时候+5
    label_index = res[2][start:end][-1] - 5
    # 获取置信度,x,y,w,h
    c,x,y,w,h,_ = label[res[0][start:end],res[1][start:end],res[2][start:end]]
    # 获取448情况下的真正坐标
    x = (x * 64) + x_id * 64
    y = (y * 64) + y_id * 64
    w = w * 448
    h = h * 448
    xmin = x - (w/2) - 10
    xmax = x + (w/2) + 10
    ymin = y - (h/2) - 10
    ymax = y + (h/2) + 10
    draw.rectangle((xmin,ymin,xmax,ymax),outline='red',width=3)
image.show()


"""

3 3
0.556 0.17333333333333378
0.7344262295081967 0.7099453551912569


# 反正求的是距离,以那个尺度下无所谓
ture = [0.556,0.17333333333333378,0.7344262295081967,0.7099453551912569]
out = [0.556,0.17333333333333378,0.7344262295081967,0.7099453551912569]
def iou(ture, out):
    x,y,w,h = ture
    px,py,pw,ph = out
    xmin,ymin,xmax,ymax = x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5
    p_xmin,p_ymin,p_xmax,p_ymax = px - pw * 0.5, py - ph * 0.5, px + pw * 0.5, py + ph * 0.5
    
    in_w = np.minimum(xmax, p_xmax) - np.maximum(xmin, p_xmin)
    in_h = np.minimum(ymax, p_ymax) - np.minimum(ymin, p_ymin)
    logical_and = np.logical_and(np.greater(in_w, 0), np.greater(in_h,0))
    inter = np.multiply(logical_and, (in_w * in_h))
    union = np.multiply(ymax-ymin, xmax - xmin) + np.multiply(p_ymax- p_ymin, p_xmax-p_xmin) - inter
    res = inter / union
    print(res)
iou(ture,out)

"""
"""
# def _cal_iou(PT, GT):
#     
#     计算IOU
#     PT,GT: [bs, x, y, w, h]
#     PT_hat, GT_hat = [bs, xmin, ymin, xmax, ymax]
#     
#     PT_true_x = PT[:,]
    
#     in_w = tf.minimum(GT_hat[:, 2], GT[:, 2]) - tf.maximum(GT_hat[:, 0], GT[:, 0])
#     in_h = tf.minimum(GT_hat[:, 3], GT[:, 3]) - tf.maximum(GT_hat[:, 1], GT[:, 1])
#     logical_and = tf.logical_and(tf.greater(in_w, 0), tf.greater(in_h,0))
#     logical_and = tf.cast(logical_and, tf.float32)
#     inter = tf.multiply(logical_and, (in_w * in_h))
#     union = tf.multiply((GT_hat[:, 3]-GT_hat[:, 1]), (GT_hat[:, 2]-GT_hat[:,0])) +\
#          tf.multiply((GT[:,3]-GT[:,1]),(GT[:,2]-GT[:,0])) - inter
#     iou = tf.divide(inter, union)
#     return iou
"""

"""
 # # 获取包含object的cell-id.
    # icl_obj_xmin_id = np.maximum((x_id - (((xmax - xmin) / 2) // 64)),0)
    # icl_obj_xmax_id = np.minimum((x_id + (((xmax - xmin) / 2) // 64)),6)
    # icl_obj_ymin_id = np.maximum((y_id - (((ymax - ymin) / 2) // 64)),0)
    # icl_obj_ymax_id = np.minimum((y_id + (((ymax - ymin) / 2) // 64)),6)
    # incloud_obj_xs = (icl_obj_xmin_id, icl_obj_xmax_id)
    # incloud_obj_ys = (icl_obj_ymin_id, icl_obj_ymax_id)
    # 整合bboexs
"""
"""
import numpy as np
import tensorflow as tf


from Loss import _losses
from GetDataInfo import FilesLoader, DataLoad
from Utels import _get_response_obj

class YoLoVgg16:
    def __init__(self,VOC_dir,batch_size):
        self.VOC_dir = VOC_dir
        self.batch_size = batch_size
        self.resize_image = 448
        self.var_list = []
        self.sess = tf.Session()
        self.w_init = tf.initializers.glorot_normal()
        self.b_init = tf.initializers.zeros()
        self.vgg_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
    
    def _leak_relu(self, data):
        data = tf.nn.leaky_relu(data, 0.1)
        return data

    def _conv2d(self,data,name,shape,strides):
        weights = tf.get_variable('W'+name, shape, initializer=self.w_init)
        bias = tf.get_variable('b'+name, (1, 1, shape[-1]), initializer=self.b_init)
        self.var_list.append(weights)
        self.var_list.append(bias)
        data = tf.add(tf.nn.conv2d(data, weights, strides, padding='SAME'), bias)
        data = self._leak_relu(data)
        return data

    def _fc_layer(self, data, name, shape, activation=None, drop=False):
        weights = tf.get_variable('W' + name, shape, initializer=self.w_init)
        bias = tf.get_variable('b' + name, (1, shape[-1]), initializer=self.b_init)
        self.var_list.append(weights)
        self.var_list.append(bias)
        data = tf.add(tf.matmul(data, weights), bias)
        if activation:
            data = activation(data)
        if drop:
            data = tf.nn.dropout(data, rate=0.5)
        return data

    def _net(self, data):
        data = self._conv2d(data, '21', (3, 3, 512, 1024), (1, 1, 1, 1))
        data = self._conv2d(data, '22', (3, 3, 1024, 1024), (1, 2, 2, 1))
        data = self._conv2d(data, '23', (3, 3, 1024, 1024), (1, 1, 1, 1))
        data = self._conv2d(data, '24', (3, 3, 1024, 1024), (1, 1, 1, 1))
        data = tf.reshape(data, (-1, 7 * 7 * 1024))
        data = self._fc_layer(data, '25', (7 * 7 * 1024, 4096), self._leak_relu, drop=False)
        data = self._fc_layer(data, '26', (4096, 7 * 7 * 30))
        data = tf.reshape(data, (-1, 7, 7, 30))
        return data
    
    def _load_data_set(self):
        files_loader = FilesLoader(VOC_dir)
        sample_train, sample_val = files_loader.get_datas()
        data_load = DataLoad(self.sess, self.batch_size, self.resize_image)
        return data_load,sample_train,sample_val

    def _running(self,samples, loader, mode, optimizer=False):
        (dataset, labels) = samples
        N = dataset.shape[0]
        next_element = loader.get_batchs(dataset,labels)
        mean_loss, count, bs = 0, 0, 0
        while True:
            try:
                image, label = self.sess.run(next_element)
                bs += image.shape[0]
                count += 1
                print('Running {} {}-{}...\r'.format(mode, bs, N), end='', flush=True)
                pre_out = self.vgg_model.predict(image)
                feed_dict = {self.datas:pre_out, self.targets:label, self.learning_rate:self.lr}
                if optimizer:
                    self.sess.run(self.Optimizer,feed_dict=feed_dict)
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                mean_loss += loss
            except tf.errors.OutOfRangeError:
                mean_loss /= count
                return mean_loss

    def yolo_go(self, epochs):
        self.datas = tf.placeholder(tf.float32, (None, 14, 14, 512))
        self.targets = tf.placeholder(tf.float32, (None, 7, 7, 25))
        self.learning_rate = tf.placeholder(tf.float32)
        self.out = self._net(self.datas)
        self.response_pre,self.new_labels = _get_response_obj(self.out, self.targets)
        self.cost = _losses(self.response_pre, self.new_labels)
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, var_list=self.var_list)

        # 加载数据
        data_load, sample_train, sample_val = self._load_data_set()
        # 保存模型
        saver = tf.train.Saver(self.var_list)

        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs+1):
            if epoch < 30:
                self.lr = 10e-5
            elif 30< epoch < 50:
                self.lr = 10e-6
            else:
                self.lr = 10e-7

            train_loss = self._running(sample_train, data_load, 'Trian', optimizer=True)
            val_loss = self._running(sample_val, data_load, 'Val', optimizer=False)
            print('{}-{} train loss:{}  val loss:{} '.format(
                epoch, epochs, train_loss, val_loss))
        save_path = saver.save(self.sess, "YOLO_VGG16/YOLO_VGG16_model.ckpt")
        print("Model saved in path: %s" % save_path)
        

if __name__ == "__main__":
    VOC_dir = '/Users/joker/jokers/DataSet/VOCdevkit_train/VOC2012'
    yolo_vgg16 = YoLoVgg16(VOC_dir, 64)
    yolo_vgg16.yolo_go(epochs=80)"""