import numpy as np
import tensorflow as tf

def _get_cell_info(o_width,o_height,resize_wh,location_list,cells_num):
    cell_num = resize_wh // cells_num # 448 // 7 = 64
    xmin,ymin,xmax,ymax = location_list
    # 等比例缩放到目标尺度
    w,h = resize_wh / o_width, resize_wh / o_height
    xmin,ymin,xmax,ymax = xmin * w, ymin * h, xmax * w, ymax * h
    center_x = (xmax + xmin) / 2
    center_y = (ymax + ymin)/ 2
    # 获取第一个cell含有x,y这个点
    x_id = int(center_x // cell_num)
    y_id = int(center_y // cell_num)
    # 得到 x和 y
    x = (center_x - cell_num * x_id) / cell_num
    y = (center_y - cell_num * y_id) / cell_num
    # 得到 w,h
    _w = (xmax - xmin) / resize_wh
    _h = (ymax - ymin) / resize_wh

    bboxes = [x,y,_w,_h]
    return  x_id, y_id, bboxes

def _get_response_obj(predict, true_label):
    """
    bs,7,7,30
    predict last dim:[c1,x1,y1,w1,h1,c2,x2,y2,w2,h2,classes...]
    """
    # 更改形状方便计算IOU
    _,s1,s2,b = predict.get_shape().as_list()
    _predict = tf.reshape(predict, (-1, s1 * s2, b))
    r_ture_label = tf.reshape(true_label, (-1, s1 * s2, b-5)) # reshape ture labels

    pre_obj1 = _predict[:,:,0:5]
    pre_obj2 = _predict[:,:,5:10]
    ture_obj = r_ture_label[:,:,0:5]
    iou_obj1 = _cal_IOU(ture_obj, pre_obj1)
    iou_obj2 = _cal_IOU(ture_obj, pre_obj2)
    # 获取b1和b2最大的iou索引.
    indeces_1 = tf.where(iou_obj1 >= iou_obj2)
    indeces_2 = tf.where(iou_obj2 > iou_obj1)
    # 确保response obj是与真实标签一一对应的.
    # response 为obj1，gather_nd: 切片操作.
    response_pre_cxywh_1 = tf.gather_nd(_predict[:,:,0:5], indeces_1)
    response_pre_classes_1 = tf.gather_nd(_predict[:,:,10:], indeces_1)
    response_pre_1 = tf.concat([response_pre_cxywh_1,response_pre_classes_1],1)
    r_ture_label_1 = tf.gather_nd(r_ture_label,indeces_1)
    # response 为 obj2
    response_pre_cxywh_2 = tf.gather_nd(_predict[:,:,5:10], indeces_2)
    response_pre_classes_2 = tf.gather_nd(_predict[:,:,10:], indeces_2)
    response_pre_2 = tf.concat([response_pre_cxywh_2,response_pre_classes_2],1)
    r_ture_label_2 = tf.gather_nd(r_ture_label,indeces_2)
    # 拼接 response_pre 和 r_ture_label
    response_pre = tf.concat([response_pre_1, response_pre_2],0)
    ture_label_ = tf.concat([r_ture_label_1, r_ture_label_2],0)
    # 更改为原来的形状
    response_pre = tf.reshape(response_pre, (-1, s1, s2, b-5))
    ture_label_ = tf.reshape(ture_label_, (-1, s1, s2, b-5))
    return response_pre, ture_label_

def _cal_IOU(GT, PT):
    """
    [bs, 49, 5]
    GT,PT last dims:[c, x, y, w, h]
    """
    Gx,Gy,Gw,Gh = GT[:,:,1], GT[:,:,2], GT[:,:,3], GT[:,:,4]
    Px,Py,Pw,Ph = PT[:,:,1], PT[:,:,2], PT[:,:,3], PT[:,:,4]

    Gxmin = tf.subtract(Gx, tf.divide(Gw, 2))
    Gxmax = tf.add(Gx, tf.divide(Gw, 2))
    Gymin = tf.subtract(Gy, tf.divide(Gh, 2))
    Gymax = tf.add(Gy, tf.divide(Gh, 2))

    Pxmin = tf.subtract(Px, tf.divide(Pw, 2))
    Pxmax = tf.add(Px, tf.divide(Pw, 2))
    Pymin = tf.subtract(Py, tf.divide(Ph, 2))
    Pymax = tf.add(Py, tf.divide(Ph, 2))

    in_w = tf.minimum(Gxmax, Pxmax) - tf.maximum(Gxmin, Pxmin)
    in_h = tf.minimum(Gymax, Pymax) - tf.minimum(Gymin, Pymin)
    logical_and = tf.logical_and(tf.greater(in_w, 0), tf.greater(in_h,0))
    logical_and = tf.cast(logical_and, tf.float32)
    inter = tf.multiply(logical_and, (in_w * in_h))
    union = tf.multiply(Gymax-Gymin, Gxmax - Gxmin) + tf.multiply(Pymax- Pymin, Pxmax-Pxmin) - inter
    iou = tf.abs(tf.div_no_nan(inter, union))
    return iou

def _NMS(bboxes, confidence, iou_thresh, max_out_put=10):
    """计算非极大抑制"""
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    xmin = tf.expand_dims(x - w /2, axis=1)
    ymin = tf.expand_dims(y - h /2, axis=1)
    xmax = tf.expand_dims(x + w /2, axis=1)
    ymax = tf.expand_dims(y + h /2, axis=1)
    bboxes = tf.concat([xmin, ymin, xmax, ymax],axis=1)
    select_indeces = tf.image.non_max_suppression(bboxes, confidence, max_out_put, iou_thresh)
    return select_indeces

def _loop(S, Acc_TP, Acc_FP,num):
    """循环TP与FP"""
    _,value = S
    if value == 1.:
        Acc_TP += 1.
    else:
        Acc_FP += 1.
    precision = Acc_TP / (Acc_TP + Acc_FP)
    recall = Acc_TP / num

    return precision,recall,Acc_TP,Acc_FP

def _Pre_MAP(response_pre, ture_label_, iou_thresh=0.3):
    """
    [c,x,y,w,h,....]
    out: [bs, 7, 7, 25]
    true_label: [bs, 7, 7 25]
    计算mAP
    """
    response_pre = tf.reshape(response_pre, (-1, 7 * 7, 25))
    ture_label_ = tf.reshape(ture_label_, (-1, 7 * 7, 25))
    response_cxywh = response_pre[:,:,0:5]
    ture_label_cxywh = ture_label_[:,:,0:5]
    iou = _cal_IOU(response_cxywh, ture_label_cxywh)
    # 获取iou大于阈值的,也就是TP部分
    where_TP = tf.where(tf.greater_equal(iou, iou_thresh))
    r_pre_TP = tf.gather_nd(response_pre[:,:,0], where_TP)
    r_pre_TP = tf.expand_dims(r_pre_TP, axis=1)
    one_TP = tf.ones_like(r_pre_TP)
    TP = tf.concat([r_pre_TP, one_TP], axis=1)
    # 获取iou小于阈值的,也就FP部分
    where_FP = tf.where(tf.less(iou, iou_thresh))
    r_pre_FP = tf.gather_nd(response_pre[:,:,0], where_FP)
    r_pre_FP = tf.expand_dims(r_pre_FP, axis=1)
    zero_FP = tf.zeros_like(r_pre_FP)
    FP = tf.concat([r_pre_FP, zero_FP], axis=1)
    TP_FP = tf.concat([TP,FP], axis=0)
    # 按照置信度排序(降序)
    Sort_TP_FP = tf.sort(TP_FP, axis=0, direction='DESCENDING')
    return Sort_TP_FP,ture_label_







if __name__ == "__main__":
    pass
    
