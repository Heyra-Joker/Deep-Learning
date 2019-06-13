import os
import numpy as np
from PIL import Image
from torchvision import transforms


class SplitData:

    def __init__(self, file_dir, Load_samples=None, Shuffle=True, test_rate=0.3):
        """
        Argus:
        -----
            file_dir (string): data set file path like "../train/"
            Load_samples (Int): Load_samples (Int): Load data number. if given None, load all samples.
            Shuffle (bool): is shuffle data set. Suggest True. default true.
            test_rate (float): split test data rate. default 0.3.
        """
        self.file_dir = file_dir
        self.test_rate = test_rate
        self.Load_samples = Load_samples
        self.Shuffle = Shuffle

    def __call__(self):

        # loading all path in current file dir.
        files_list = os.listdir(self.file_dir)

        if self.Shuffle:
            np.random.shuffle(files_list)

        if self.Load_samples:
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

        return train_files, test_files, train_samples, test_samples


def generate_train(data, batch_size):
    """
    Argus:
    ------
    data (list): include training data set files path.
    batch_size (Int): batch size loaded.

    Return:
    -------
        yield (X, y), it's a generate.
    """
    m = len(data)
    N = m // batch_size
    N = np.maximum(N, 1)

    np.random.shuffle(data)

    while True:

        for n in range(N):
            X = []
            y = []
            bs_data = data[n * batch_size:(n + 1) * batch_size]

            for file in bs_data:

                classes = file.split('/')[-1].split('.', 1)[0]
                if classes == 'cat':
                    label = 0
                else:
                    label = 1

                image = Image.open(file)
                image = image.resize((224, 224))
                image = np.array(image)
                image = np.divide(image, 255)

                X.append(image)
                y.append(label)

            else:
                X = np.array(X)
                X = np.pad(X, pad_width=((0, 0), (1, 2), (1, 2), (0, 0)), mode='constant')
                y = np.array(y)
                yield (X, y)


def generate_test(data, batch_size):
    """
        Argus:
        ------
        data (list): include training data set files path.
        batch_size (Int): batch size loaded.

        Return:
        -------
            yield (X, y), it's a generate.
    """

    m = len(data)
    N = m // batch_size
    N = np.maximum(N, 1)

    np.random.shuffle(data)

    while True:
        for n in range(N):
            X = []
            y = []
            bs_data = data[n * batch_size:(n + 1) * batch_size]
            for file in bs_data:

                classes = file.split('/')[-1].split('.', 1)[0]
                if classes == 'cat':
                    label = 0
                else:
                    label = 1

                image = Image.open(file)
                image = image.resize((224, 224))
                image = np.array(image)
                image = np.divide(image, 255)

                X.append(image)
                y.append(label)

            else:
                X = np.array(X)
                X = np.pad(X, pad_width=((0, 0), (1, 2), (1, 2), (0, 0)), mode='constant')
                y = np.array(y)
                yield (X, y)


class Crop:

    def __init__(self, file_path):
        """
        Argus:
        -----
            file_path (list):   include testing data set files path.
        """
        self.file_path = file_path

    def __call__(self):
        image = Image.open(self.file_path)
        image = image.resize((256, 256))
        # get five pictures.
        five = transforms.FiveCrop(size=(224, 224))
        five_img = five(image)
        # transpose image of left and right.
        five_img_transpose = [np.array(img.transpose(Image.FLIP_LEFT_RIGHT)) for img in five_img]
        five_img = [np.array(img) for img in five_img]

        # stack two array. result shape (10,...)
        Img = np.vstack((five_img_transpose, five_img)) / 255
        Img = np.pad(Img, pad_width=((0, 0), (1, 2), (1, 2), (0, 0)), mode='constant')
        return Img


if __name__ == "__main__":

    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    split_data = SplitData(file_dir, Load_samples=10, test_rate=0.3)
    train_files, test_files, train_samples, test_samples = split_data()

    generate = generate_test(test_files, 100)

    m = len(test_files)
    N = m // 100
    N = np.maximum(N, 1)
    for i in range(N):
        a1, a2 = next(generate)
        print(a1.shape, a2)

    # cat = 'cat.jpg'
    # crop = Crop(cat)
    # Img = crop()
    # print(Img.shape)
