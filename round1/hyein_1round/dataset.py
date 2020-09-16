#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:58:31 2020

@author: hihyun
"""
from pathlib import Path
import shutil
from tempfile import mkdtemp
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from warnings import warn
from nsml.constants import DATASET_PATH
CLASSES = ['normal', 'monotone', 'screenshot', 'unknown', 'unlabeled']
UNLABELED = -1
# From baseline code
def prepare(base_dir: Path):
    def _initialize_directory(dataset: str) -> None:
        dataset_path = base_dir / dataset
        dataset_path.mkdir()
        for c in CLASSES:
            (dataset_path / c).mkdir()
    def _rearrange(dataset: str) -> None:
        output_dir = base_dir / dataset
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        for _, row in metadata.iterrows():
            if row['annotation'] == UNLABELED:
                row['annotation'] = 4
            src = src_dir / 'train_data' / row['filename']
            if not src.exists():
                raise FileNotFoundError
            dst = output_dir / CLASSES[row['annotation']] / row['filename']
            if dst.exists():
                warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
            else:
                shutil.copy(src=src, dst=dst)
    dataset = 'train'
    _initialize_directory(dataset)
    _rearrange(dataset)
def preprocess_train_info(base_dir: Path, sup: bool=True):
    prepare(base_dir)
    dataset_info = {
        'img_path': [],
        'label': []
    }
    for label, kind in enumerate(CLASSES):
        paths = [path for path in Path(base_dir / 'train').glob(f'{kind}/*.*') if path.suffix not in ['.gif', '.GIF']]
        for path in paths:
            dataset_info['img_path'].append(str(path))
            dataset_info['label'].append(label)
    print(dataset_info['label'].count(0))
    print(dataset_info['label'].count(1))
    print(dataset_info['label'].count(2))
    print(dataset_info['label'].count(3))

    dataset_info = pd.DataFrame(dataset_info).sample(frac=1.)
        # Remove unlabeled samples
    if not sup:
        unlabel_info=dataset_info[dataset_info.label == 4].reset_index(drop=True) #labeled info
    dataset_info = dataset_info[dataset_info.label != 4].reset_index(drop=True) #drop unlabeled info
    train_info, valid_info = train_test_split(dataset_info, test_size=0.2,random_state=24)
    return train_info, valid_info,unlabel_info
def preprocess_test_info(test_dir: str):
    dataset_info = {
        'img_path': []
    }
    paths = [path for path in (Path(test_dir) / 'test_data').glob('*.*') if path.suffix not in ['.gif', '.GIF']]
    for path in paths:
        dataset_info['img_path'].append(str(path))
    dataset_info = pd.DataFrame(dataset_info)
    return dataset_info
class SpamDataset(Dataset):
    def __init__(self, img_paths: list, labels: list,
                 num_classes: int = 4, tfms=None, test=False):
        self.img_paths = img_paths
        self.labels = labels
        self.num_classes = num_classes
        self.tfms = tfms
        self.test = test
        if not self.test:#only for training sampling
            self.weight=[0]*len(self.img_paths)
            self.w_d={0:0.0063,1:0.6035,2:0.3050,3:0.0851}
            #weight list
            for idx, l in enumerate(self.labels):
                self.weight[idx]=self.w_d[l]
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.tfms:
            image = self.tfms(image)
        if self.test:
            return image, img_path.split('/')[-1]
        else:
            return image, label
    def __len__(self):
        return len(self.img_paths)
    def get_labels(self):
        return list(self.labels)
class ulDataset(Dataset):
    def __init__(self, img_paths: list,
                 num_classes: int = 4, tfms=None, test=False):
        self.img_paths = img_paths
        self.num_classes = num_classes
        self.tfms = tfms
        self.test = test
        """
        if not self.test:#only for training sampling
            self.weight=[0]*len(self.img_paths)
            self.w_d={0:0.0063,1:6035,2:0.3050,3:0.0851}
            #weight list
            for idx, l in enumerate(self.labels):
                self.weight[idx]=self.w_d[l]
        """
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        if self.tfms:
            image = self.tfms(image)

        return image
    def __len__(self):
        return len(self.img_paths)
    def get_labels(self):
        return list(self.labels)
if __name__ == '__main__':
    from torchvision.transforms import transforms
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    """ Train & Valid """
    base_dir = Path(mkdtemp())
    train_info, valid_info = preprocess_train_info(base_dir, sup=True)
    train_dataset = SpamDataset(train_info.img_path.values,
                                train_info.label.values,
                                tfms=tfms)
    valid_dataset = SpamDataset(valid_info.img_path.values,
                                valid_info.label.values,
                                tfms=tfms)
    print(train_info, valid_info)
    print(next(iter(train_dataset)))
    """ Test at bind_model.infer(test_dir, **kwargs) """
    # test_info = preprocess_test_info(test_dir)
    # test_dataset = SpamDataset(test_info.img_path.values,
    #                            test_info.index.values,
    #                            tfms=tfms,
    #                            test=True)
