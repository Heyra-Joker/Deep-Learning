import keras
from keras.models import Sequential
import keras.layers as Layers

from loadData_Keras import SplitData,generator


def ZFNet_Sequential(rate,m_classes):
    model = Sequential()
    #Conv1
    model.add(
        Layers.Conv2D(filters=96,kernel_size=(7,7),strides=(2,2),padding='valid',
                      activation='relu',input_shape=(225,225,3))
    )
    model.add(
        Layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same')
    )
    #Conv2
    model.add(
        Layers.Conv2D(filters=256,kernel_size=(5,5),strides=(2,2),padding='valid',activation='relu')
    )
    model.add(
        Layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same')
    )
    #Conv3
    model.add(
        Layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')
    )
    #Conv4
    model.add(
        Layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')
    )
    #Conv5
    model.add(
        Layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')
    )
    model.add(
        Layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')
    )
    #flatten
    model.add(
        Layers.Flatten()
    )
    #Fc6
    model.add(
        Layers.Dense(units=4096,activation='relu')
    )
    model.add(
        Layers.Dropout(rate=rate)
    )
    #Fc7
    model.add(
        Layers.Dense(units=4096,activation='relu')
    )
    model.add(
        Layers.Dropout(rate)
    )
    #out
    model.add(
        Layers.Dense(units=m_classes,activation='sigmoid')
    )
    print(model.summary())
    return model



def ZFNet(file_dir,lr,epochs=32,Load_samples=100,test_rate=0.3,batch_size=50,rate=0.5,m_classes=1,save_model=None):


    split_data = SplitData(file_dir, Load_samples=Load_samples, test_rate=test_rate)
    (train_files, train_samples), (test_files, test_samples) = split_data()

    N_train = train_samples // batch_size
    N_test = test_samples // batch_size

    train_generator = generator(train_files,batch_size)
    test_generator = generator(test_files,batch_size)


    model = ZFNet_Sequential(rate,m_classes)
    #loss
    loss = keras.losses.binary_crossentropy
    #optimizer
    optim = keras.optimizers.RMSprop(lr=lr)
    #complile
    model.compile(optimizer=optim,loss=loss,metrics=['accuracy'])
    #fit of generator
    model.fit_generator(train_generator,steps_per_epoch=N_train,epochs=epochs,
                        validation_data=test_generator,validation_steps=N_test)
    #save
    if save_model:
        print('Saved in {}'.format(save_model))
        model.save(save_model)


if __name__ == "__main__":
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    save_model = 'ZFNet_Keras.h5'
    zf = ZFNet(file_dir,lr=1e-4,Load_samples=100,test_rate=0.3,batch_size=10,rate=0.5)




