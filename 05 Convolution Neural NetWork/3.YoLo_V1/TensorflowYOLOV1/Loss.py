import tensorflow as tf
def _losses(predict, ture_label):
    lambda_coord = 5.
    lambda_noobj = .5
    _,s1,s2,b = predict.get_shape().as_list()
    reshape_predict = tf.reshape(predict, (-1, s1 * s2, b))
    reshape_labels = tf.reshape(ture_label, (-1, s1 * s2, b))
    # 找出包含obj的cell,也就是标签中c>0的(或者说c=1)
    where_obj = tf.greater(reshape_labels[:,:,0],0.)
    indices_obj = tf.where(where_obj)
    predict_obj = tf.gather_nd(reshape_predict, indices_obj)
    labels_obj = tf.gather_nd(reshape_labels, indices_obj)
    # 找不包含obj的cell
    where_noobj = tf.equal(reshape_labels[:,:,0],0.)
    indices_noobj = tf.where(where_noobj)
    predict_noobj = tf.gather_nd(reshape_predict, indices_noobj)
    labels_noobj = tf.gather_nd(reshape_labels, indices_noobj)
    # 计算置信度loss
    # 包含obj的置信度.
    p_c_obj = predict_obj[:,0]
    l_c_obj = labels_obj[:,0]
    loss_c_obj = tf.reduce_sum(tf.square(p_c_obj - l_c_obj))
    # 不包含obj的置信度
    p_c_noobj = predict_noobj[:,0]
    l_c_noobj = labels_noobj[:,0]
    loss_c_noobj = tf.reduce_sum(tf.square(p_c_noobj - l_c_noobj)) * lambda_noobj
    loss_c = loss_c_noobj + loss_c_obj
    # 计算x,y
    # 包含obj的x,y
    p_x_obj = predict_obj[:,1]
    l_x_obj = labels_obj[:,1]
    p_y_obj = predict_obj[:,2]
    l_y_obj = labels_obj[:,2]
    loss_xy = tf.reduce_sum(tf.add(tf.square(p_x_obj - l_x_obj), tf.square(p_y_obj - l_y_obj)))
    loss_xy = loss_xy * lambda_coord
    # 计算w,h
    # 包含obj的w,h
    p_w_obj = predict_obj[:,3]
    l_w_obj = labels_obj[:,3]
    p_h_obj = predict_obj[:,4]
    l_h_obj = labels_obj[:,4]
    # 保证前期平稳度过,所以这里使用maximum,方式w,h为负数
    loss_w = tf.square(tf.sqrt(tf.maximum(p_w_obj, 1e-10)) - tf.sqrt(l_w_obj))
    loss_h = tf.square(tf.sqrt(tf.maximum(p_h_obj, 1e-10)) - tf.sqrt(l_h_obj))
    loss_wh = tf.reduce_sum(tf.add(loss_w, loss_h)) * lambda_coord
    # 计算类别,包含obj的
    p_classes_obj = predict_obj[:, 5:]
    l_classes_obj = labels_obj[:, 5:]
    loss_classes = tf.reduce_sum(tf.square(p_classes_obj - l_classes_obj))
    # 合并最终loss
    loss = tf.reduce_mean(loss_c + loss_xy + loss_wh + loss_classes)
    return loss


if __name__ == "__main__":
    predict = tf.random.truncated_normal((2, 7, 7, 25))
    labels = tf.random.uniform((2, 7, 7, 25))
    res = _losses(predict, labels)
    sess = tf.Session()
    res = sess.run(res)
    print(res)
    print(res.shape)
