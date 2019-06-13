import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


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


class Load(Dataset):

    def __init__(self, data_files, transform=None):

        self.data_files = data_files
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):

        file_path = self.data_files[item]

        classes = file_path.split('/')[-1].split('.', 1)[0]
        if classes == 'cat':
            label = 0
        else:
            label = 1

        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = np.array(img)

        img = np.divide(img, 255)

        img = np.pad(img, pad_width=((1, 2), (1, 2), (0, 0)), mode='constant')

        sample = (img, np.array([label]))

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:

    def __call__(self, sample):
        img, label = sample
        n_w, n_h, n_c = img.shape

        img = img.reshape((n_c, n_h, n_w))
        img = torch.FloatTensor(img)
        label = torch.IntTensor(label)

        sample = (img, label)

        return sample


if __name__ == "__main__":
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    split_data = SplitData(file_dir, Load_samples=10, test_rate=0.3)
    train_files, test_files, train_samples, test_samples = split_data()

    load = Load(train_files,transform=ToTensor())
    train_loader = DataLoader(load,batch_size=2,shuffle=True,num_workers=2)

    for img,label in train_loader:
        img = img[0].numpy()
        img = img.reshape((227,227,3))
        plt.imshow(img)
        plt.title(label[0].item())
        plt.show()

