import keras
import  numpy as np
from keras.models import Sequential
import keras.layers as LayerS

from LoadDataset import  *



def AlexNet(file_dir,Load_samples,test_rate,drop_rate,lr,batch_size,save_path=None):
    """
    Argus:
    ------
        file_dir (string): data set file path. like "../train"
        Load_samples (Int): Load data number. if given None, load all samples.
        test_rate (float): split test data rate.
        drop_rate (float): dropout layers parameter, dropout_rate = 1 - keep prob.
        lr (float): learning rate with RMSProp.
        batch_size (Int): batch size.
        save_path (string): save model path, if not equal None.

    Return:
    ------
        model (class): Sequential model.
    """

    # Split training set and testing set.
    split_data = SplitData(file_dir,Load_samples=Load_samples,Shuffle=True,test_rate=test_rate)
    train_files, test_files, train_samples, test_samples = split_data()

    print('Train Samples: {}'.format(train_samples))
    print('Test Samples: {}'.format(test_samples))

    m_train = len(train_files)
    N_train = int(np.maximum(m_train // batch_size,1))


    m_test = len(test_files)
    N_test = int(np.maximum(m_test // batch_size,1))

    # Create Model.
    model = Sequential()
    # CONV1
    model.add(
        LayerS.Conv2D(filters=96,kernel_size=(11,11),
                      strides=(4,4),padding='valid',input_shape=(227,227,3)))
    model.add(
        LayerS.Activation('relu'))
    model.add(
        LayerS.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    # Normal

    # CONV2
    model.add(
        LayerS.Conv2D(filters=256,kernel_size=(5,5),
                      strides=(1,1),padding='same'))
    model.add(
        LayerS.Activation('relu'))
    model.add(
        LayerS.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    # Normal2

    # CONV3
    model.add(
        LayerS.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(
        LayerS.Activation('relu'))

    # CONV4
    model.add(
        LayerS.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(
        LayerS.Activation('relu'))

    # CONV5
    model.add(
        LayerS.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(
        LayerS.Activation('relu'))
    model.add(
        LayerS.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))

    # FC6
    model.add(
        LayerS.Flatten())
    model.add(
        LayerS.Dense(units=4096))
    model.add(
        LayerS.Activation('relu'))
    model.add(
        LayerS.Dropout(rate=drop_rate)
    )

    # FC7
    model.add(
        LayerS.Dense(units=4096))
    model.add(
        LayerS.Activation('relu'))
    model.add(
        LayerS.Dropout(rate=drop_rate))

    # FC8
    model.add(
        LayerS.Dense(units=1000))
    model.add(
        LayerS.Activation('relu'))
    model.add(
        LayerS.Dropout(rate=drop_rate))

    # output
    model.add(
        LayerS.Dense(units=1))

    model.add(
        LayerS.Activation('sigmoid'))

    # create loss.
    loss = keras.losses.binary_crossentropy
    # create optimizer.
    optimizer = keras.optimizers.RMSprop(lr=lr)

    # compile model.
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    # fit model at generator.
    model.fit_generator(generate_train(train_files,batch_size),
                        steps_per_epoch=N_train,epochs=13,validation_data=generate_test(test_files,batch_size),
                        validation_steps=N_test)

    # save model.
    if save_path:
        print('Saving Model In {}'.format(save_path))
        model.save(save_path)
    return model



def Predict(file_path,model_path):
    """
    Argus:
    ------
        file_path (string): predict picture path.
        model_path (string): Sequential model path.

    Return:
    ------
        None.

    """
    crop = Crop(file_path)
    Img = crop()
    model = keras.models.load_model(model_path)
    predict = model.predict(Img)
    predict = np.round(predict)
    dog_rate = np.sum(predict) /  predict.size
    cat_rate = 1 - dog_rate

    print('The Picture is dog :{} %'.format(dog_rate * 100))
    print('The Picture is cat :{} %'.format(cat_rate * 100))



if __name__ == "__main__":

    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    save_path = 'Alexnet_Keras_model.h5'
    AlexNet(file_dir, Load_samples=None, test_rate=0.3, drop_rate=0.3, lr=1e-4, batch_size=100,save_path=save_path)

    file_path ='cat.jpeg'
    model_path = 'Alexnet_Keras_model.h5'
    Predict(file_path,model_path)