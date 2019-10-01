import json
import numpy as np
from cv2 import cv2 as cv


def read_json(json_path):
    with open(json_path,'r', encoding='utf8') as f:
        classes = json.loads(f.read())
        return classes

def FileShow(image_path,label):
    resize_wh = 448
    single_cell = 64
    font=cv.FONT_HERSHEY_SIMPLEX
    classes = read_json('flip_classes_id.json')

    image = cv.imread(image_path)
    o_h,o_w = image.shape[:2]
    
    n_w,n_h = o_w / resize_wh, o_h / resize_wh

    nozeros = np.nonzero(label)
    N = nozeros[0].shape[0] // 6
    for i in range(N):
        start = i * 6
        end = (i + 1) * 6
        x_id,y_id = nozeros[0][start:end][0], nozeros[1][start:end][0]
        cls_index = nozeros[2][start:end][-1] - 5
        c,x,y,w,h,_ = label[nozeros[0][start:end],nozeros[1][start:end],nozeros[2][start:end]]
        x = (x * single_cell) + x_id * single_cell
        y = (y * single_cell) + y_id * single_cell
        w = w * resize_wh
        h = h * resize_wh
        xmin = int((x - (w/2))*n_w)
        xmax = int((x + (w/2))*n_w)
        ymin = int((y - (h/2))*n_h)
        ymax = int((y + (h/2))*n_h)
        image = cv.rectangle(image, (xmin, ymin),(xmax, ymax),(255, 0, 0), 2)
        text = str(c) + ':' + classes[str(cls_index)]
        image = cv.putText(image,text,(xmin,ymin),font,.5,(0,0,255),2)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def PredictShow(image_path , out, index, classes):
    """out: 原始网络产生的值"""
    font=cv.FONT_HERSHEY_SIMPLEX
    image = cv.imread(image_path)
    image = cv.resize(image, (448,448))
    m = index.shape[0]
    for i in range(m):
        x_id, y_id, = index[i]
        cls_index = np.argmax(out[i, 0:20])
        C = np.max(out[i, 0:20])
        # 如果NMS传入的不是20个最大的,这里的20号索引要删除 ！！！
        x, y, w, h = out[i, 21:25]
        # 获取448尺度下的x,y位置
        x = (x * 64) + x_id * 64 
        y = (y * 64) + y_id * 64
        # 获取448尺度下的w,h
        w = w * 448
        h = h * 448
        # 获取左上和右下的坐标点
        xmin = int(x - w / 2 )
        ymin = int(y - h / 2 )
        xmax = int(x + w / 2 )
        ymax = int(y + h / 2 )
        # 获得类别值
        classes_name = classes[str(int(cls_index))]
        # 绘制边框
        image = cv.rectangle(image, (xmin, ymin),(xmax, ymax),(0, 255, 0), 2)
        text = str(round(C,2)) + ':' + classes_name
        image = cv.putText(image,text,(xmin,ymin),font,.5,(0,0,255),1)

    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def CreameShow(image , out, index, classes):
    o_h,o_w = image.shape[:2]
    n_w,n_h = o_w / 448, o_h / 448
    font=cv.FONT_HERSHEY_SIMPLEX
    m = index.shape[0]
    for i in range(m):
        x_id, y_id, = index[i]
        cls_index = np.argmax(out[i, 0:20])
        C = np.max(out[i, 0:20])
        # 如果NMS传入的不是20个最大的,这里的20号索引要删除 ！！！
        x, y, w, h = out[i, 21:25]
        # 获取448尺度下的x,y位置
        x = (x * 64) + x_id * 64 
        y = (y * 64) + y_id * 64
        # 获取448尺度下的w,h
        w = w * 448
        h = h * 448
        # 获取左上和右下的坐标点
        xmin = int((x - (w/2))*n_w)
        xmax = int((x + (w/2))*n_w)
        ymin = int((y - (h/2))*n_h)
        ymax = int((y + (h/2))*n_h)
        # 获得类别值
        classes_name = classes[str(int(cls_index))]
        # 绘制边框
        image = cv.rectangle(image, (xmin, ymin),(xmax, ymax),(0, 255, 0), 2)
        text = str(round(C,2)) + ':' + classes_name
        image = cv.putText(image,text,(xmin,ymin),font,.5,(0,0,255),1)
    return image
       








