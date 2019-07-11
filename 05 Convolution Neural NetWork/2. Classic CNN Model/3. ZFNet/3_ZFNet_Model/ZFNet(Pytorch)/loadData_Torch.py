import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


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

        return train_files, test_files


class LoadData(Dataset):

    def __init__(self, data_files,transform=None):
        self.data_files = data_files
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        file = self.data_files[item]
        classes = file.split('/')[-1].split('.',1)[0]
        if classes == 'cat':
            label = 1
        else:
            label = 0
        image = Image.open(file)
        image = image.resize((224,224))
        image = np.array(image)

        sample = (image,label)
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor:


    def __call__(self,sample):
        image,label = sample
        w,h,c = image.shape
        image = image.reshape((c,h,w))
        image = torch.FloatTensor(image)
        image = torch.div(image,255)
        label = torch.LongTensor([label])
        label = torch.squeeze(label)
        sample = (image,label)

        return sample

class Normal:
    def __call__(self,sample):
        image, label = sample
        normal = transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        image = normal(image)
        sample = (image,label)
        return sample


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Pytorch Using {}'.format(device))

    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'

    split_data = SplitData(file_dir, Load_samples=100, test_rate=0.3)
    train_files, test_files = split_data()

    compose = transforms.Compose([ToTensor(),Normal()])
    loader = LoadData(train_files,compose)

    train_loader = DataLoader(loader,batch_size=50,shuffle=True,num_workers=2)
    print(len(train_loader.dataset))
    for epoch in range(2):
        for image,label in train_loader:
            print(image.size())
            print(label.size())
            img = image[0].numpy()
            img = img.reshape((224,224,3))
            lab = label[0].item()
            plt.imshow(img)
            plt.title(lab)
            plt.show()
        print('hahah')



