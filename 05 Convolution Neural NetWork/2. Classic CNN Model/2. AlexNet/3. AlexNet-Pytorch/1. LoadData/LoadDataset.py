import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class LoadCatsVsDogs(Dataset):
    """
    加载数据集并继承Dataset类
    """
    def __init__(self, file_dir, n_h, n_w, transform=None):
        """
        初始化类
        Args:
        ----
            file_dir (string): 数据集图片的目录.
            n_h (int): resize 指定的图片高
            n_w (int): resize 指定的
            transform (callable, optional): 选择一个transform 来应用一个样本.
        """
        self.file_dir = file_dir
        self.list_dirs = os.listdir(self.file_dir)
        self.n_h = n_h
        self.n_w = n_w
        self.transform = transform

    def __len__(self):
        """
        返回样本总数
        """
        return len(self.list_dirs)
    
    def __getitem__(self,idx):
        """
        getitem是类Dataset中的一个迭代器.
        """
        image_name = self.list_dirs[idx]
        image_path = os.path.join(self.file_dir,image_name)
        # 图片名形式:{classes}.N.jpg,所以使用split拆分.
        classes = image_name.split('.',1)[0]
        if classes == 'cat':
            label = 0
        else:
            label = 1

        # resize
        image = Image.open(image_path)
        image = image.resize((self.n_w,self.n_h))
        image = np.array(image)

        sample = {'image':image,'label':label}
        if self.transform:
            
            sample = self.transform(sample)

        return sample
        
class Normal_:
    """
    Normal Image
    """

    def __call__(self,sample):
        image,label = sample['image'],sample['label']
        n_w,n_h,n_c = image.shape
        flatten = image.flatten()
        mean = np.mean(flatten)
        stddev = np.std(flatten)
        divi = np.maximum(stddev, 1./np.sqrt(n_w * n_h * n_c))
        normal = (flatten - mean) / divi
        image = normal.reshape((n_w,n_h,n_c))

        return {'image':image,'label':label}

class ToTensor:
    """
    Change to Tensor
    """
    def __call__(self,sample):
        image,label = sample['image'],sample['label']
        n_w,n_h,n_c = image.shape
        # numpy image: W x H x C
        # torch image: C X H X W
        image = image.reshape((n_c,n_h,n_w))
        image = torch.from_numpy(image).float()
        label = torch.FloatTensor([label])
        return {'image':image,'label':label}




class Crop:
    """
    Crop Picture.
    """
    def __init__(self, file_path, size):
        """
        file_path (string): 测试样本的路径
        size (tuple): 裁剪的大小,是一个tuple,其中包含(H,W)
        """
        self.file_path = file_path
        # H x W
        self.five_crop = transforms.FiveCrop(size)

    def __call__(self):
        
        image = Image.open(self.file_path)
        image = image.resize((256,256))
        Five_image = self.five_crop(image) # return PIL type
        transpose_ = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in Five_image]
        transpose_array = [np.array(img) for img in transpose_]
        img_array = [np.array(img) for img in Five_image]

        image_array = np.vstack((transpose_array,img_array))
        return image_array


if __name__ == "__main__":
    n_h,n_w = 224,224
    file_dir='/Users/huwang/Joker/Data_Set/catVSdot/train'

    CatsDogs_dataset = LoadCatsVsDogs(file_dir=file_dir,n_h=n_h,n_w=n_w)
    fig = plt.figure()
    for i in range(3):
        sample = CatsDogs_dataset[i]
        ax = plt.subplot(1,3,i+1)
        # Automatically adjust subplot parameters to give specified padding.
        plt.tight_layout()
        ax.set_title('Sample label:{}'.format(sample['label']))
        ax.axis('off') # trun of axis
        plt.imshow(sample['image'])
    plt.show()


    CatsDogs_dataset = LoadCatsVsDogs(file_dir=file_dir,n_h=n_h,n_w=n_w,transform=ToTensor())
    for i in range(3):
        sample = CatsDogs_dataset[i]

        print(sample['image'].size(),sample['label'].size())

    CatsDogs_dataset = LoadCatsVsDogs(file_dir=file_dir,n_h=n_h,n_w=n_w,transform=ToTensor())
    dataloader = DataLoader(CatsDogs_dataset, batch_size=4, shuffle = True, num_workers=2)

    for index,sample in enumerate(dataloader):
        print(sample['image'].size(),sample['label'].size())
        if index == 4:
            break

    file_path = 'cat.jpg'
    corp = Crop(file_path,size=(224,224))
    fig = plt.figure(figsize=(20,30))

    for i in range(5):
        image = corp()[i]
        ax = plt.subplot(1,5,i+1)
        plt.tight_layout()
        ax.set_title('Sample index:{}'.format(i))
        ax.axis('off') # trun of axis
        plt.imshow(image)
    plt.show()

    fig = plt.figure(figsize=(20,30))
    for i in range(5,10):
        image = corp()[i]
        ax = plt.subplot(1,5,i-4)
        plt.tight_layout()
        ax.set_title('Sample index:{}'.format(i))
        ax.axis('off') # trun of axis
        plt.imshow(image)
    plt.show()

