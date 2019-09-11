import os
import numpy as np
from lxml import etree
from sklearn.model_selection import train_test_split

from Classes import classes_name

"""
Load data file information.
It's have two routes:
    1. regression:
    -------------
        in this code, use per-class-regression(PCR).
        routes: load_files ==> __load_files_bbox ==> get_info_bbox.
    2. classification:
    ------------------
        routes: load_files ==> __load_files_labels ==> get_info_labels.
    
    Then: shuffle data set.
    Finally: rely on split_rate to decompose split data set. split_rate(train,val,test).
    At last: rely on test_file_save_path to save test files. 
"""

class LoadFiles_:
    def __init__(self,Annotation_dir,split_rate,test_file_save_path,target_mode='labels',bbox_classes=None):
        self.Annotation = Annotation_dir
        self.split_rate = split_rate
        self.target_mode = target_mode
        self.bbox_classes = bbox_classes
        self.test_file_save_path = test_file_save_path

        self.IMAGES = []
        self.LABELS = []

    def save_test_file(self,images,labels,prefix):
        if not os.path.exists(self.test_file_save_path):
            os.mkdir(self.test_file_save_path)
        np.save(os.path.join(self.test_file_save_path,prefix+'_test_images.npy'),images)
        np.save(os.path.join(self.test_file_save_path,prefix+'_test_labels.npy'),labels)
        print('[+] Test files save in %s'%self.test_file_save_path)


    def get_info_labels(self,tree):
        name = tree.xpath('//annotation/object/name/text()')[0]
        if '-' in name:
            name = name.replace('-','_')
        return name

    def get_info_bbox(self,tree):
        height = int(tree.xpath('//annotation/size/height/text()')[0])
        width = int(tree.xpath('//annotation/size/width/text()')[0])
        xmin = int(tree.xpath('//annotation/object/bndbox/xmin/text()')[0])
        ymin = int(tree.xpath('//annotation/object/bndbox/ymin/text()')[0])
        xmax = int(tree.xpath('//annotation/object/bndbox/xmax/text()')[0])
        ymax = int(tree.xpath('//annotation/object/bndbox/ymax/text()')[0])
        # get percent of position. -1和1对于位置而言都是一样的.
        xmin= np.log(np.maximum((xmin/ width),1e-3))
        ymin = np.log(np.maximum((ymin/ height),1e-3))
        xmax = np.log(np.maximum((xmax/ width),1e-3))
        ymax = np.log(np.maximum((ymax/ height),1e-3))
        label = [xmin,ymin,xmax,ymax]
        return label
        

    def shuffle(self):
        """
        Shuffle dataset.
        """
        IMAGES = np.array(self.IMAGES)
        LABELS = np.array(self.LABELS)
        permu_ = np.random.permutation(len(self.IMAGES))
        IMAGES = IMAGES[permu_,...]
        LABELS = LABELS[permu_,...]
        return IMAGES,LABELS

    def split(self,IMAGES,LABELS,split_rate):
        """
        split_rate: train:val:test or train:test
        """
        if round(sum(split_rate)) != 1:
            raise ValueError('[-] the split rate sum not equal 1 !')
        if len(split_rate) == 3:
            _,val_rate,test_rate = split_rate
            N = IMAGES.shape[0]
            # val sample
            NV = int(N * val_rate)
            val_images = IMAGES[0:NV,...]
            val_labels = LABELS[0:NV,...]
            # test sample
            NT = int(N * test_rate)
            test_images = IMAGES[NV:NV + NT,...]
            test_labels = LABELS[NV:NV + NT,...]
            # train sample
            train_images = IMAGES[NV + NT:,...]
            train_labels = LABELS[NV + NT:,...]

            samples_train = (train_images,train_labels)
            samples_val = (val_images,val_labels)
            samples_test = (test_images,test_labels)
            
            return samples_train,samples_val,samples_test
    
    def __load_files_labels(self):
        Annotation_dirs = os.listdir(self.Annotation)
        for dir_ in Annotation_dirs:
            dir_path = os.path.join(self.Annotation,dir_)
            class_files = os.listdir(dir_path)
            for file in class_files:
                file_path = os.path.join(dir_path,file)
                # get image path
                image_path = file_path.replace('Annotation','Images') + '.jpg'
                tree = etree.parse(file_path)
                name = self.get_info_labels(tree)
                label = classes_name[name]
                self.IMAGES.append(image_path)
                self.LABELS.append(label)
    
    def __load_files_bbox(self):
        Annotation_dirs = os.listdir(self.Annotation)
        if self.bbox_classes not  in Annotation_dirs:
            raise NameError('[-] The %s not in dirs,please check dir name.'%self.bbox_classes)
        else:
            index = Annotation_dirs.index(self.bbox_classes) # find target classes index.
            dir_ = Annotation_dirs[index]
            dir_path = os.path.join(self.Annotation,dir_)
            class_files = os.listdir(dir_path)
            for file in class_files:
                file_path = os.path.join(dir_path,file)
                # get image path
                image_path = file_path.replace('Annotation','Images') + '.jpg'
                # get bbox
                tree = etree.parse(file_path)
                labels = self.get_info_bbox(tree)
                self.IMAGES.append(image_path)
                self.LABELS.append(labels)
            
    def load_files(self):
        """
        Load mudel files.
        """
        if self.target_mode not in ['labels','bboxs']:
            raise ValueError('[-] Specify except labels/bboxs not %s'%self.target_mode)

        if self.target_mode == "labels":
            self.__load_files_labels()
        else:
            if self.bbox_classes:
                self.__load_files_bbox()
            else:
                raise KeyError('[-] Please specify bbox classes!')

        IMAGES,LABELS = self.shuffle()
        samples_train,samples_val,samples_test = self.split(IMAGES,LABELS,self.split_rate)
        print('[+] The total samples is %d;train:%d;val:%d;test:%d'%
        (len(self.IMAGES),samples_train[0].shape[0],samples_val[0].shape[0],samples_test[0].shape[0]))

        # save test file
        self.save_test_file(samples_test[0],samples_test[1],self.target_mode)
        return samples_train,samples_val
        

                
                
if __name__ == "__main__":
    from PIL import Image,ImageDraw
    Annotation_dir = '/Users/joker/jokers/DataSet/stanford-dogs-dataset/Annotation'
    test_file_save_path = 'TEST_FILES'
    """
    loadfiles_ = LoadFiles_(Annotation_dir=Annotation_dir,
    split_rate=(0.8,0.1,0.1),test_file_save_path=test_file_save_path,target_mode='labels')

    # labels 
    samples_train,samples_val = loadfiles_.load_files()
    print(samples_train[1][0])
    print(samples_train[0][0])
    img = Image.open(samples_train[0][0])
    img.show()
    """
    """
    # bboxs
    # African_hunting_dog
    loadfiles_ = LoadFiles_(Annotation_dir,(0.7,0.2,0.1),
    test_file_save_path,target_mode='bboxs',bbox_classes='n02116738-African_hunting_dog')
    samples_train,samples_val = loadfiles_.load_files()
    img = Image.open(samples_train[0][0])
    img = img.resize((224,224))
    draw = ImageDraw.Draw(img)
    label = samples_train[1][0] * 224
    xmin,ymin,xmax,ymax = label
    draw.rectangle([xmin,ymin,xmax,ymax],outline='red',width=3)
    img.show()"""