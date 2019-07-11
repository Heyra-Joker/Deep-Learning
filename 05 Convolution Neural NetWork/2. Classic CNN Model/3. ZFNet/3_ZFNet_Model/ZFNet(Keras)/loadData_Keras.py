import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SplitData:
    """
    Split training data and testing data.
    """

    def __init__(self, file_dir, Load_samples=None, Shuffle=True, test_rate=0.3):
        """
        :param file_dir (string): data dir. like '../train' it's a directory.
        :param Load_samples (int,None): loading Number,if equal None loading all files,default None.
        :param Shuffle (bool): is shuffle original files. suggest True.
        :param test_rate (float): test data rate.
        """
        self.file_dir = file_dir
        self.test_rate = test_rate
        self.Load_samples = Load_samples
        self.Shuffle = Shuffle

    def __call__(self):
        """
        :return train_files, test_files (list): include train files path and test files path.
        """
        # loading all path in current file dir.
        files_list = os.listdir(self.file_dir)

        if self.Shuffle:
            np.random.shuffle(files_list)

        if self.Load_samples is not None:
            files_list = files_list[:self.Load_samples]

        # split data
        len_ = len(files_list)
        test_index = int(np.floor(len_ * self.test_rate))
        test_files = files_list[:test_index]
        train_files = files_list[test_index:]

        # join path
        test_files = [os.path.join(self.file_dir, file) for file in test_files]
        train_files = [os.path.join(self.file_dir, file) for file in train_files]

        train_samples, test_samples = len_ - test_index, test_index
        print('Training Samples:{},Testing Samples:{}'.format(train_samples, test_samples))

        return (train_files,train_samples), (test_files,test_samples)


def generator(files,batch_size):

    m = len(files)
    N = np.maximum(1, m//batch_size)

    np.random.shuffle(files)

    while True:
        for n in range(N):
            images = []
            labels = []
            datas = files[n*batch_size:(n+1)*batch_size]
            for data in datas:
                classes = data.split('/')[-1].split('.',1)[0]
                labels.append([1 if classes=="cat" else 0])

                image = Image.open(data)
                image = image.resize((224,224))
                image = np.array(image)
                w,h,c = image.shape
                image = image.reshape((h,w,c))
                image = np.divide(image,255)
                image = np.pad(image,pad_width=((1,0),(1,0),(0,0)),mode='constant')
                images.append(image)

            images = np.array(images)
            labels = np.array(labels)
            yield images,labels


if __name__ == "__main__":
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    split_data = SplitData(file_dir, Load_samples=100, test_rate=0.3)
    (train_files, train_samples), (test_files, test_samples) = split_data()

    for i in range(2):
        images,labels = next(generator(train_files, 50))
        print(images.shape)
        print(labels.shape)
        plt.imshow(images[0])
        plt.title(labels[0][0])
        plt.show()
