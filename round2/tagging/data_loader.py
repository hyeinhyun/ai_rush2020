import os

import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0
        self.min = 100000
        self.max = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)
            if self.min>val:
                self.min=val
            if self.max<val:
                self.max=val
        else:
            pass


class TagImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, root_dir: str, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.size=AverageMeter()
        self.weight=[0]*len(self.data_frame)
        self.w_d={0:0.1,1:0.1,2:1.5,3:0.1,4:0.01}
        for idx,l in enumerate(list(self.data_frame['answer'])):
            self.weight[idx]=self.w_d[l]
            
        """
        print(len(self.data_frame['answer'].unique()))
        print(len(self.data_frame['category_1'].unique()))
        print(len(self.data_frame['category_2'].unique()))
        print(len(self.data_frame['category_3'].unique()))
        print(len(self.data_frame['category_4'].unique()))
        
        ########카테고리 별 answer 분포
        print("0&출산/육아{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['출산/육아'])).sum()))
        print("0&패션잡화{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['패션잡화'])).sum()))
        print("0&식품{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['식품'])).sum()))
        print("0&패션의류{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['패션의류'])).sum()))
        print("0&가구/인테리어{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['가구/인테리어'])).sum()))
        print("0&화장품/미용{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['화장품/미용'])).sum()))
        print("0&스포츠/레저{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['스포츠/레저'])).sum()))
        print("0&생활/건강{}".format((self.data_frame["answer"].isin([0])&self.data_frame["category_1"].isin(['생활/건강'])).sum()))
        
        print("1&출산/육아{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['출산/육아'])).sum()))
        print("1&패션잡화{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['패션잡화'])).sum()))
        print("1&식품{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['식품'])).sum()))
        print("1&패션의류{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['패션의류'])).sum()))
        print("1&가구/인테리어{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['가구/인테리어'])).sum()))
        print("1&화장품/미용{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['화장품/미용'])).sum()))
        print("1&스포츠/레저{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['스포츠/레저'])).sum()))
        print("1&생활/건강{}".format((self.data_frame["answer"].isin([1])&self.data_frame["category_1"].isin(['생활/건강'])).sum()))
        
        print("2&출산/육아{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['출산/육아'])).sum()))
        print("2&패션잡화{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['패션잡화'])).sum()))
        print("2&식품{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['식품'])).sum()))
        print("2&패션의류{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['패션의류'])).sum()))
        print("2&가구/인테리어{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['가구/인테리어'])).sum()))
        print("2&화장품/미용{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['화장품/미용'])).sum()))
        print("2&스포츠/레저{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['스포츠/레저'])).sum()))
        print("2&생활/건강{}".format((self.data_frame["answer"].isin([2])&self.data_frame["category_1"].isin(['생활/건강'])).sum()))
        
        print("3&출산/육아{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['출산/육아'])).sum()))
        print("3&패션잡화{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['패션잡화'])).sum()))
        print("3&식품{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['식품'])).sum()))
        print("3&패션의류{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['패션의류'])).sum()))
        print("3&가구/인테리어{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['가구/인테리어'])).sum()))
        print("3&화장품/미용{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['화장품/미용'])).sum()))
        print("3&스포츠/레저{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['스포츠/레저'])).sum()))
        print("3&생활/건강{}".format((self.data_frame["answer"].isin([3])&self.data_frame["category_1"].isin(['생활/건강'])).sum()))
        
        print("4&출산/육아{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['출산/육아'])).sum()))
        print("4&패션잡화{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['패션잡화'])).sum()))
        print("4&식품{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['식품'])).sum()))
        print("4&패션의류{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['패션의류'])).sum()))
        print("4&가구/인테리어{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['가구/인테리어'])).sum()))
        print("4&화장품/미용{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['화장품/미용'])).sum()))
        print("4&스포츠/레저{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['스포츠/레저'])).sum()))
        print("4&생활/건강{}".format((self.data_frame["answer"].isin([4])&self.data_frame["category_1"].isin(['생활/건강'])).sum()))
        """

        #print(self.data_frame.columns)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data_frame.iloc[idx]['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        cat1=self.data_frame.iloc[idx]['category_1']
        tag_name = self.data_frame.iloc[idx]['answer']
        sample['label'] = tag_name
        sample['image_name'] = img_name
        sample['cat1']=cat1
        return sample


class TagImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = [img for img in os.listdir(self.root_dir) if not img.startswith('.')]
        self.data_list.remove('test_input')
        #meta=self.root_dir +'/test_input'
        #print(pd.read_csv(meta))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        sample['image_name'] = img_name
        return sample