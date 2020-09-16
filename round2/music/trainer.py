import os
import json
import random
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct
from adamp import AdamP
from torch.optim.optimizer import Optimizer
import math
import random

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import librosa
from sklearn.metrics import f1_score
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from sklearn.model_selection import KFold
import nsml
#custom
from model import CRNN, CRNN2, VGG, LSTM,AuxSkipAttention
from pretrainedmodels import se_resnet152
from resize_random import RandomResizedCrop
# from mixer import RandomMixer,SigmoidConcatMixer,AddMixer
from cnn_14 import Cnn14,ResNet38
from cbam import resnet50_cbam
# criterion = nn.CrossEntropyLoss(reduction='mean')
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def mono_to_color(X1,X2,X3, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X1, X2, X3])

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


# def spec_augment(spec: np.ndarray, num_mask=2,
#                  freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
#     spec = spec.copy()
#     for i in range(num_mask):
#         all_frames_num, all_freqs_num = spec.shape
#         freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
#
#         num_freqs_to_mask = int(freq_percentage * all_freqs_num)
#         f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
#         f0 = int(f0)
#         spec[:, f0:f0 + num_freqs_to_mask] = 0
#
#         time_percentage = random.uniform(0.0, time_masking_max_percentage)
#
#         num_frames_to_mask = int(time_percentage * all_frames_num)
#         t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
#         t0 = int(t0)
#         spec[t0:t0 + num_frames_to_mask, :] = 0
#
#     return spec
# cv.setNumThreads(0)
def get_transforms(train, size,
                   wrap_pad_prob=0.5,
                   resize_scale=(0.8, 1.0),
                   resize_ratio=(1.7, 2.3),
                   resize_prob=0.33,
                   spec_num_mask=2,
                   spec_freq_masking=0.15,
                   spec_time_masking=0.20,
                   spec_prob=0.5):
    if train:
        transforms = Compose([
            RandomCrop(size),
            UseWithProb(
                # Random resize crop helps a lot, but I can't explain why ¯\_(ツ)_/¯
                RandomResizedCrop(scale=(0.8, 1.0), ratio=(1.7, 2.3)),
                prob=0.33
            ),
            UseWithProb(SpecAugment(num_mask=spec_num_mask,
                                    freq_masking=spec_freq_masking,
                                    time_masking=spec_time_masking), spec_prob),
            # HorizontalFlip(),
            ImageToTensor()

        ])
    else:
        transforms = Compose([
            CenterCrop(size),
            ImageToTensor()

        ])
    return transforms

def sectorfinder(mel, winsize = 1200):
    if mel.shape[1] < winsize:
        return mel

    maxp = 0
    maxi = -1
    powers = sum(mel)
    #print(powers.shape)
    power = sum(powers[0:winsize])
    for i in range(mel.shape[1]-winsize):
        power += powers[i+winsize]
        power -= powers[i]

        if power > maxp:
            maxp = power
            maxi = i

    return mel[:,maxi:maxi+winsize]
def image_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def gauss_noise(image, sigma_sq):
    h, w = image.shape
    gauss = np.random.normal(0, sigma_sq, (h, w))
    gauss = gauss.reshape(h, w)
    image = image + gauss
    return image


# Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num  = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec


class SpecAugment:
    def __init__(self,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20):
        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def __call__(self, image):
        return spec_augment(image,
                            self.num_mask,
                            self.freq_masking,
                            self.time_masking,
                            image.min())


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class Flip:
    def __init__(self, flip_code):
        assert flip_code == 0 or flip_code == 1
        self.flip_code = flip_code

    def __call__(self, image):
        image = cv.flip(image, self.flip_code)
        return image


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class VerticalFlip(Flip):
    def __init__(self):
        super().__init__(0)


class GaussNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, image):
        if self.sigma_sq > 0.0:
            image = gauss_noise(image,
                                np.random.uniform(0, self.sigma_sq))
        return image


class RandomGaussianBlur:
    '''Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    '''
    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image):
        kernel_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv.GaussianBlur(image, kernel_size, self.sigma_x)
        return blured_image


class ImageToTensor:
    def __call__(self, image):
        delta = librosa.feature.delta(image)
        mfcc = dct(image, axis=0, type=2, norm="ortho")
        accelerate = librosa.feature.delta(image, order=2)
        image = np.stack([image, accelerate, delta], axis=0)
        # image = np.stack([image],axis=0)
        image = image.astype(np.float32) / 100
        image = torch.from_numpy(image)
        return image


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        start = random.randint(0, signal.shape[1] - self.size)
        return signal[:, start: start + self.size]


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):

        if signal.shape[1] > self.size:
            start = (signal.shape[1] - self.size) // 2
            # return signal[:,:self.size]
            return signal[:, start: start + self.size]
        else:
            return signal


class PadToSize:
    def __init__(self, size, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = size
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            padding = self.size - signal.shape[1]
            offset = padding // 2
            pad_width = ((0, 0), (offset, padding - offset))
            if self.mode == 'constant':
                signal = np.pad(signal, pad_width,
                                'constant', constant_values=signal.min())
            else:
                signal = np.pad(signal, pad_width, 'wrap')
        return signal

class AddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target,rnd_image,rnd_target):
        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        #target = (1 - alpha) * target + alpha * rnd_target
        return image, alpha

class MusicDataset(Dataset):
    def __init__(self, config, dataset_root, train=True):
        self.dataset_name = config['dataset_name']  # i.e. q1
        self.input_length = config['input_length']
        self.train = train

        self.mel_dir = os.path.join(dataset_root, 'train_data', 'mel_spectrogram')
        self.label_file = os.path.join(dataset_root, 'train_label')
        with open(self.label_file) as f:
            self.train_labels = json.load(f)
        label_types = {'q1': 'station_name',
                       'q2': 'mood_tag',
                       'q3': 'genre'}

        self.label_type = label_types[self.dataset_name]
        self.label_map = self.create_label_map()
        self.n_classes = len(self.label_map)

        self.cvn=5
        self.n_cv=0
        self.id_list=[i for i in range(len(self.train_labels['track_index']))]


        #make self cv =5
        kf = KFold(n_splits=self.cvn,shuffle=True,random_state=0)
        g=kf.split(self.id_list)
        cur_g=next(g)





        self.l_valid=cur_g[1]
        self.n_valid=len(self.l_valid)
        self.l_train=cur_g[0]

        print(self.l_valid)
        # self.l_train_tmp=self.l_train.copy()
        # for i in self.l_train_tmp:
        #     if self.train_labels[self.label_type][str(i)]==0:
        #         pass
        #     else:
        #         self.l_train=np.append(self.l_train,i)
        # print('Over sampling : ',len(self.l_train))
        self.n_train = len(self.l_train)



        # self.n_valid = len(self.train_labels['track_index']) // 10
        # self.n_train = len(self.train_labels['track_index']) - self.n_valid



    def create_label_map(self):
        label_map = {}

        if self.dataset_name in ['q1', 'q3']:
            for idx, label in self.train_labels[self.label_type].items(): # 'station_name'
                if label not in label_map:
                    label_map[label] = len(label_map)
        else:
            for idx, label_list in self.train_labels[self.label_type].items():
                for label in label_list:
                    if label not in label_map:
                        label_map[label] = len(label_map)

        return label_map

    def __getitem__(self, idx):
        data_idx = str(self.l_train[idx])
        if not self.train:
            data_idx = str(self.l_valid[idx])
        # data_idx = str(idx)
        # if not self.train:
        #     data_idx = str(idx + self.n_train)

        track_name = self.train_labels['track_index'][data_idx]
        mel_path = os.path.join(self.mel_dir, '{}.npy'.format(track_name))
        mel = np.load(mel_path)[0]
        mel = np.log10(1 + 10000 * mel) * 10



        # mel = dct(mel, axis=0, type=2, norm="ortho")
        # mel = sectorfinder(mel)#1200
        # mel = mel[:,:1200]
        # if mel.shape[1]<3600:
        #     padding=3600-len(mel)
        #     offset=padding//2
        #     mel=np.pad(mel,((0,0),(0,padding)),'constant')
        # else:
        #     mel=mel[:,:3600]

        # mel = mel[:, 800:800+self.input_length]
        # #mel = mono_to_color(mel)
        # mel = mono_to_color(mel,mel,mel)
        # mel1 = mel[:, :1200]
        # mel2 = mel[:, 1200:2400]
        # mel3 = mel[:, 2400:3600]
        label = self.train_labels[self.label_type][data_idx]
        if self.dataset_name in ['q1', 'q3']:
            labels = self.label_map[label]
        else:
            label_idx = [self.label_map[l] for l in label]
            labels = np.zeros(self.n_classes, dtype=np.float32)
            labels[label_idx] = 1

        if self.train:


            # mel1=spec_augment(mel1)
            # mel2=spec_augment(mel2)
            # mel3=spec_augment(mel3)
            tfms1=get_transforms(self.train,1200)
            # tfms2=get_transforms(self.train,1200)
            # tfms3=get_transforms(self.train,1200)

            mel=tfms1(mel)

            # print(mel.shape)
            # mel=mel_t
            # mel2=tfms2(mel)
            # mel3=tfms3(mel)
            # mel = mono_to_color(mel1, mel2, mel3)


        else:
            tfms=get_transforms(self.train,1200)
            mel=tfms(mel)
            # # mel = mono_to_color(mel, mel, mel)
            # mel1=tfms(mel)#center crop
            # mel = mono_to_color(mel1, mel1, mel1)
            # mel1 = mel[:, :self.input_length]
            # mel2 = mel[:, int((mel.shape)[1] * 0.5) - 600:int((mel.shape)[1] * 0.5) + 600]
            # mel3 = mel[:, (mel.shape)[1] - self.input_length:]
            # tfms1=get_transforms(self.train,1200)
            # mel=tfms1(mel)


            # mel = mono_to_color(mel1, mel2, mel3)
            #mel = mono_to_color(mel1, mel1, mel1)
            # mel = mel2



        # librosa의 power to db와의 차이는 결국 np.log10()한 뒤에 10을 곱해주느냐 혹은 여기 있는걸로 하느냐의 차이
        # normalization 개념은 아니고, 오히려 더 뚜렷하게 구분하는데 도움을 준다고 보면 되겠다. 가장 엄밀히 말하면 그다지 차이가 없다고 보는 게 정확할 듯 하다.
        # 밖에다 곱연산하는것보다는 확실히 더 뚜렷하게 구분한다고 보면 될 듯. 1을 더하는건 그냥 양수로 만들기 위해서이기 때문에 신경 쓸 필요 없다.
        # 더 자세하게 특징을 구별하고 싶다면. mel에 곱해져 있는 계수를 올릴 것

        #mfcc = dct(mel, axis=0, type = 2, norm="ortho") # 여기서 [:n] 을 붙이면 n만큼의 계수를 뽑아내게 된다. dct 특성상 아마 맨 앞에 있는 계수들부터

        # mel = mel[:, int((mel.shape)[1]*0.5)-600:int((mel.shape)[1]*0.5)+600]


        return mel, labels

    def __len__(self):
        return self.n_train if self.train else self.n_valid


class TestMusicDataset(Dataset):
    def __init__(self, config, dataset_root):
        self.dataset_name = config['dataset_name']  # i.e. q1

        self.meta_dir = os.path.join(dataset_root, 'test_data', 'meta')
        self.mel_dir = os.path.join(dataset_root, 'test_data', 'mel_spectrogram')

        meta_path = os.path.join(self.meta_dir, '{}_test.json'.format(self.dataset_name))
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.input_length = config['input_length']
        self.n_classes = 100 if self.dataset_name == 'q2' else 4

    def __getitem__(self, idx): #테스트용인거 참고
        data_idx = str(idx)

        track_name = self.meta['track_index'][data_idx]
        mel_path = os.path.join(self.mel_dir, '{}.npy'.format(track_name))
        mel = np.load(mel_path)[0]
        mel = np.log10(1 + 10000 * mel) * 10
        # mel = dct(mel, axis=0, type=2, norm="ortho")

        # mel = mel[:,:1200]
        # mel = sectorfinder(mel)#1200

        tfms1 = get_transforms(False, 1200)
        mel = tfms1(mel)

        # if mel.shape[1]<3600:
        #     padding=3600-len(mel)
        #     offset=padding//2
        #     mel=np.pad(mel,((0,0),(0,padding)),'constant')
        # else:
        #     mel=mel[:,:3600]
        # # #
        # # # # mel = mel[:, 800:800+self.input_length]
        # # # # #mel = mono_to_color(mel)
        # # # # mel = mono_to_color(mel,mel,mel)
        # mel1 = mel[:, :1200]
        # mel2 = mel[:, 1200:2400]
        # mel3 = mel[:, 2400:3600]
        # # # # mel = mel[:, 800:800+self.input_length]
        # # # # #mel = mono_to_color(mel)
        # # # # mel = mono_to_color(mel,mel,mel)
        # # #
        # # # # mel1 = mel[:, :self.input_length]
        # # # # mel2 = mel[:, int((mel.shape)[1]*0.5)-600:int((mel.shape)[1]*0.5)+600]
        # # # # mel3 = mel[:, (mel.shape)[1]-self.input_length:]
        # # # # mel = mono_to_color(mel1, mel2, mel3)
        # mel = mono_to_color(mel1, mel2, mel3)
        # # # # mel = mel2


        #mfcc = dct(mel, axis=0, type = 2, norm="ortho")
        #mel = mel[:, int((mel.shape)[1]*0.5)-600:int((mel.shape)[1]*0.5)+600]
        #mel = mel[:, :self.input_length]

        return mel, data_idx

    def __len__(self):
        return len(self.meta['track_index'])


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class BasicNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_classes = config['n_classes']

        self._extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((4, 8))
        )

        self._classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=256, out_features=n_classes))
        self.apply(init_weights)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self._extractor(x)
        x = x.view(x.size(0), -1)
        score = self._classifier(x)
        return score


def select_optimizer(model, config):
    args = config['optimizer']
    lr = args['lr']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    nesterov = args['nesterov']
    name = args['name']

    if name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        return optimizer

    if name == 'AdamP':
        optimizer = AdamP(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, nesterov=True)
        return optimizer
    if name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer
    if name == 'RAdam':
        optimizer = RAdam(model.parameters(),lr=lr,weight_decay=weight_decay)
        return optimizer
    if name == 'AdamW':
        optimizer = AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
        return optimizer


def select_scheduler(optimizer, config):
    args = config['schedule']
    factor = args['factor']
    patience = args['patience']

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    return scheduler


def accuracy_(pred, target):
    _, predicted_max_idx = pred.max(dim=1)
    n_correct = predicted_max_idx.eq(target).sum().item()
    return n_correct / len(target)


def f1_score_(pred, target, threshold=0.5):
    pred = np.array(pred.cpu() > threshold, dtype=float)
    return f1_score(target.cpu(), pred, average='micro')


class Trainer:
    def __init__(self, config, mode):
        """
        mode: train(run), test(submit)
        """
        self.device = config['device']
        self.dataset_name = config['dataset_name']
        self.config = config
        self.maxim = 0


        if mode == 'train':
            batch_size = config['batch_size']
            self.train_dataset = MusicDataset(config, config['dataset_root'], train=True)
            self.valid_dataset = MusicDataset(config, config['dataset_root'], train=False)
            self.label_map = self.train_dataset.label_map
            #self.tfms=transforms.Compose([transforms.RandomHorizontalFlip()])

            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False)

        if self.dataset_name in ['q1', 'q3']:
            config['n_classes'] = 4
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.act = torch.nn.functional.log_softmax
            self.act_kwargs = {'dim': 1}
            self.measure_name = 'accuracy'
            self.measure_fn = accuracy_
        else:
            config['n_classes'] = 100
            self.criterion = nn.BCELoss(reduction='none')
            self.act = torch.sigmoid
            self.act_kwargs = {}
            self.measure_name = 'f1_score'
            self.measure_fn = f1_score_

        # self.model = BasicNet(config).to(self.device)
        # self.model = CRNN2().cuda()
        #self.model = VGG().cuda()
        # self.model = AuxSkipAttention(4).cuda()
        # self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=4).cuda()
        # self.model=Cnn14(4)
        # self.model=ResNet38(4)
        # self.model = se_resnet152(num_classes=1000, pretrained='imagenet')
        # self.model.last_linear = nn.Sequential(
        #     nn.Dropout(p=0.4),
        #     nn.Linear(in_features=1000, out_features=5, bias=True)
        # )
        # self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(p=0.4),
        #     nn.Linear(in_features=2048, out_features=4, bias=True)
        # )
        # self.model = torchvision.models.vgg19_bn(pretrained=True)
        # self.model.classifier[6] = nn.Sequential(nn.Dropout(p=0.4),
        # nn.Linear(in_features=4096, out_features=4, bias=True))
        self.model=resnet50_cbam(pretrained=True)
        # self.model = torchvision.models.resnet152(pretrained=True)
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(p=0.4),
        #     nn.Linear(in_features=2048, out_features=4, bias=True)
        # )

        # self.model = LSTM().cuda()
        self.model=self.model.cuda()
        self.optimizer = select_optimizer(self.model, config)
        self.scheduler = select_scheduler(self.optimizer, config)

        self.iter = config['iter']
        self.val_iter = config['val_iter']
        self.save_iter = config['save_iter']


    def run_batch(self, batch, train):
        x, y = batch
        x, y = x.to(self.device,dtype=torch.float), y.to(self.device)

        if train:
            self.optimizer.zero_grad()
            y_ = self.model(x)
            y_ = self.act(y_, **self.act_kwargs)
            loss = torch.mean(self.criterion(y_, y))
            loss.backward()
            self.optimizer.step()
        else:
            y_ = self.model(x)
            y_ = self.act(y_.detach(), **self.act_kwargs)
            loss = torch.mean(self.criterion(y_, y))

        loss = loss.item()
        measure = self.measure_fn(y_, y)
        batch_size = y.size(0)

        return loss * batch_size, measure * batch_size, batch_size

    def run_train(self, epoch=None):
        if epoch is not None:
            print(f'Training on epoch {epoch}')

        data_loader = self.train_loader
        self.model.train()

        total_loss = 0
        total_measure = 0
        total_cnt = 0

        n_batch = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            batch_status = self.run_batch(batch, train=True)
            loss, measure, batch_size = batch_status

            total_loss += loss
            total_measure += measure
            total_cnt += batch_size

            if batch_idx % (n_batch // 10) == 0:
                print(batch_idx, '/', n_batch)

        status = {'train__loss': total_loss / total_cnt,
                  'train__{}'.format(self.measure_name): total_measure / total_cnt}
        return status

    def run_valid(self, epoch=None):
        if epoch is not None:
            print(f'Validation on epoch {epoch}')

        data_loader = self.valid_loader
        self.model.eval()

        total_loss = 0
        total_measure = 0
        total_cnt = 0
        for batch_idx, batch in enumerate(data_loader):
            batch_status = self.run_batch(batch, train=False)
            loss, measure, batch_size = batch_status

            total_loss += loss
            total_measure += measure
            total_cnt += batch_size

        score = total_measure / total_cnt

        status = {'valid__loss': total_loss / total_cnt,
                  'valid__{}'.format(self.measure_name): score}

        if score > self.maxim: # 
            self.maxim = score
            self.save('9999')
        
        return status

    def run_evaluation(self, test_dir):
        """
        Predicted Labels should be a list of labels / label_lists
        """
        dataset = TestMusicDataset(self.config, test_dir)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.model.eval()

        idx2label = {v: k for k, v in self.label_map.items()}

        predicted_prob_list = []
        predicted_labels = []
        for x, data_idx in loader:
            x = x.to(self.device,dtype=torch.float)
            y_ = self.model(x)

            if self.dataset_name in ['q1', 'q3']:
                predicted_probs, predicted_max_idx = y_.max(dim=1)
                predicted_labels += list(predicted_max_idx)
            else:
                threshold = 0.5
                over_threshold = np.array(y_.cpu() > threshold, dtype=float)
                label_idx_list = [np.where(labels == 1)[0].tolist() for labels in over_threshold]
                predicted_labels += label_idx_list

        if self.dataset_name in ['q1', 'q3']:
            predicted_labels = [idx2label[label_idx.item()] for label_idx in predicted_labels]
        else:
            predicted_labels = [[idx2label[label_idx] for label_idx in label_idx_list] for label_idx_list in
                                predicted_labels]

        return predicted_labels

    def run(self):
        for epoch in range(self.iter):
            epoch_status = self.run_train(epoch)

            if epoch % self.val_iter == 0:
                self.report(epoch, epoch_status)

                valid_status = self.run_valid(epoch)
                self.report(epoch, valid_status)

                self.scheduler.step(valid_status['valid__loss'])

            if epoch % self.save_iter == 0:
                self.save(epoch)
                

    def save(self, epoch):
        nsml.save(epoch)
        print(f'Saved model at epoch {epoch}')

    def report(self, epoch, status):
        print(status)
        nsml.report(summary=True, scope=locals(), step=epoch, **status)
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

