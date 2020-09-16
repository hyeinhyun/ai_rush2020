#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:19:17 2020

@author: hihyun
"""

from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
import os
from argparse import ArgumentParser
from tempfile import mkdtemp
import pandas as pd
from sklearn.metrics import classification_report
#torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

#torch
from torch import nn, optim
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
#torch.utils.data.WeightedRandomSampler
#system
import sys
import argparse
import json
import math
import typing
from tqdm import tqdm
import numpy as np
from pathlib import Path
from adamp import AdamP

#model
from efficientnet_pytorch import EfficientNet
#nsml
import nsml

#custom
#from data_folder import folder
from dataset import preprocess_train_info, SpamDataset,preprocess_test_info,ulDataset
from model import Resnet
from model import Densenet
from model import wrn
from model import Res50,Rex100,Res152
from randaugment import RandAugmentMC
from resnext import build_resnext
#augment
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from cutout import Cutout


def bind_model(model):
    def save(dirname, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        model.load_state_dict(checkpoint['model'])

    def infer(test_dir, **kwargs):########test infer 부분!
        return evaluate(model,test_dir)

    nsml.bind(save=save, load=load, infer=infer)
#for scheduler
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)
#for log
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

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)
        else:
            pass




def full_train(tr_dloader,val_dloader,model,epochs):

    for child in list(model.children())[:-2]:
        for param in child.parameters():
                param.requires_grad = False
    model.train()
    ##prepare 
    optimizer=AdamP(model._fc.parameters(), 0.0001)
    #warmup=0
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer)

    #iteration=68479//32
    #total_steps=300*iteration
    #scheduler = get_cosine_schedule_with_warmup(optimizer,warmup*iteration,total_steps)
    criterion = nn.CrossEntropyLoss()
    save_idx = 0
    best_f1=0
    for epoch in range(epochs):
        #scheduler.step()
        loss_tr=AverageMeter()
        for data in tqdm(tr_dloader,total=len(tr_dloader)):
            img=data[0].cuda()
            labels=data[1].cuda()

            optimizer.zero_grad()
            out=model(img)
            loss=criterion (out,labels)
            loss_tr.update(loss.item(),img.shape[0])
            loss.backward()
            optimizer.step()
        print('train loss : {}'.format(loss_tr.avg))

            
        #val check
        model.eval()
        label_li = torch.zeros((0, )).cuda() # initialize
        preds = torch.zeros((0, )).cuda() 
        loss_val=AverageMeter()
        with torch.no_grad():
            for data in tqdm(val_dloader,total=len(val_dloader)):
                img=data[0].cuda()
                label=data[1].cuda()
                out=model(img)
                loss=criterion (out,label)
                loss_val.update(loss.item(),img.shape[0])
                output_prob = F.softmax(out, dim=1)
                _,pred=torch.max(output_prob,1)#max value
                label=label.reshape((label.shape[0], )).float()
                label_li=torch.cat((label_li,label))
                pred=pred.reshape((pred.shape[0], )).float()
                preds=torch.cat((preds,pred))
            cls_report=classification_report(label_li.cpu(),preds.cpu(),labels=[1,2,3],output_dict=True)#1,2,3만 포커스  #계속 이친구로 best잡을지 고민.
            numbers=[cls_report['1']['f1-score'],cls_report['2']['f1-score'],cls_report['3']['f1-score']]
            print(numbers[0])
            print(numbers[1])
            print(numbers[2])

            g_mean=np.exp(np.mean(np.log(numbers)))

        scheduler.step(loss_val.avg)
        nsml.save(save_idx)
        print("{}/{} : fl ; {} / val_loss : {}".format(epoch, epochs,g_mean,loss_val.avg))
        if best_f1<g_mean:#클수록 좋단말이다!
            best_f1=g_mean
            nsml.save("best")#always save best
            torch.save(model.state_dict(),'full_best.pt')
            
        save_idx+=1
def full_train2(tr_dloader,val_dloader,model,epochs):
    for child in list(model.children())[:-2]:
        for param in child.parameters():
                param.requires_grad = True

    model.train()
    ##prepare 
    optimizer=AdamP(model.parameters(), 0.0001)
    #warmup=0
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer)

    #iteration=68479//32
    #total_steps=300*iteration
    #scheduler = get_cosine_schedule_with_warmup(optimizer,warmup*iteration,total_steps)
    criterion = nn.CrossEntropyLoss()
    save_idx = 0
    best_f1=0
    for epoch in range(epochs):
        #scheduler.step()
        loss_tr=AverageMeter()
        for data in tqdm(tr_dloader,total=len(tr_dloader)):
            img=data[0].cuda()
            labels=data[1].cuda()

            optimizer.zero_grad()
            out=model(img)
            loss=criterion (out,labels)
            loss_tr.update(loss.item(),img.shape[0])
            loss.backward()
            optimizer.step()
        print('train loss : {}'.format(loss_tr.avg))

            
        #val check
        model.eval()
        label_li = torch.zeros((0, )).cuda() # initialize
        preds = torch.zeros((0, )).cuda() 
        loss_val=AverageMeter()
        with torch.no_grad():
            for data in tqdm(val_dloader,total=len(val_dloader)):
                img=data[0].cuda()
                label=data[1].cuda()
                out=model(img)
                loss=criterion (out,label)
                loss_val.update(loss.item(),img.shape[0])
                output_prob = F.softmax(out, dim=1)
                _,pred=torch.max(output_prob,1)#max value
                label=label.reshape((label.shape[0], )).float()
                label_li=torch.cat((label_li,label))
                pred=pred.reshape((pred.shape[0], )).float()
                preds=torch.cat((preds,pred))
            cls_report=classification_report(label_li.cpu(),preds.cpu(),labels=[1,2,3],output_dict=True)#1,2,3만 포커스  #계속 이친구로 best잡을지 고민.
            numbers=[cls_report['1']['f1-score'],cls_report['2']['f1-score'],cls_report['3']['f1-score']]
            print(numbers[0])
            print(numbers[1])
            print(numbers[2])

            g_mean=np.exp(np.mean(np.log(numbers)))

        scheduler.step(loss_val.avg)
        nsml.save(save_idx)
        print("{}/{} : fl ; {} / val_loss : {}".format(epoch, epochs,g_mean,loss_val.avg))
        if best_f1<g_mean:#클수록 좋단말이다!
            best_f1=g_mean
            nsml.save("best")#always save best
            torch.save(model.state_dict(),'full_best.pt')
            
        save_idx+=1


def fine_train(tr_dloader,val_dloader,ul_dloader,model,epochs):
    for child in list(model.children())[:-2]:
        for param in child.parameters():
                param.requires_grad = True

    model.train()
    ##prepare 
    optimizer=AdamP(model.parameters(), 0.0001)
    #optimizer = optim.Adam(model.parameters(), 0.0001) #optimizer/ scheduler dict도 챙겨야함
    #optimizer=optim.SGD(model.parameters(),0.0001,momentum=0.9,nesterov=True)
    #warmup=0
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer)

    #iteration=68479//32
    #total_steps=300*iteration
    #scheduler = get_cosine_schedule_with_warmup(optimizer,warmup*iteration,total_steps)
    criterion = nn.CrossEntropyLoss()
    save_idx = 0
    best_f1=0
    for epoch in range(epochs):
        #scheduler.step()
        loss_tr=AverageMeter()

            
        #fine tune
        train_loader=zip(tr_dloader,ul_dloader)
        for (data_l,data_u) in train_loader:
        #for idx,(data_u) in enumerate(ul_dloader):
        #    data_l=list(tr_dloader)[idx%len(tr_dloader)]
            img=data_l[0]
            labels=data_l[1].cuda()
            ul_w=data_u[0]
            ul_s=data_u[1]
            batch_size=img.shape[0]
            inputs=torch.cat((img,ul_w,ul_s)).cuda()
            optimizer.zero_grad()
            out=model(inputs)
            out_l=out[:batch_size]
            out_w,out_s=out[batch_size:].chunk(2)
            del out
            loss_l=criterion (out_l,labels)
            
            p_l=torch.softmax(out_w, dim=-1)
            max_probs, targets_u = torch.max(p_l, dim=-1)
            mask=max_probs.ge(0.95).float()
            
            loss_u = (F.cross_entropy(out_s, targets_u,
                              reduction='none') * mask).mean()
            
            
            loss=loss_l+loss_u
            loss_tr.update(loss.item(),inputs.shape[0])
            loss.backward()
            optimizer.step()
        print('train loss : {}'.format(loss_tr.avg))
            
        #val check
        model.eval()
        label_li = torch.zeros((0, )).cuda() # initialize
        preds = torch.zeros((0, )).cuda() 
        loss_val=AverageMeter()
        with torch.no_grad():
            for data in tqdm(val_dloader,total=len(val_dloader)):
                img=data[0].cuda()
                label=data[1].cuda()
                out=model(img)
                loss=criterion (out,label)
                loss_val.update(loss.item(),img.shape[0])
                output_prob = F.softmax(out, dim=1)
                _,pred=torch.max(output_prob,1)#max value
                label=label.reshape((label.shape[0], )).float()
                label_li=torch.cat((label_li,label))
                pred=pred.reshape((pred.shape[0], )).float()
                preds=torch.cat((preds,pred))
            cls_report=classification_report(label_li.cpu(),preds.cpu(),labels=[0,1,2,3],output_dict=True)#1,2,3만 포커스  #계속 이친구로 best잡을지 고민.
            numbers=[cls_report['1']['f1-score'],cls_report['2']['f1-score'],cls_report['3']['f1-score']]
            print(numbers[0])
            print(numbers[1])
            print(numbers[2])
            print("total acc : {}".format(cls_report['accuracy']))

            g_mean=np.exp(np.mean(np.log(numbers)))

        scheduler.step(loss_val.avg)
        nsml.save("{}_fm".format(save_idx))
        print("{}/{} : fl ; {} / val_loss : {}".format(epoch, epochs,g_mean,loss_val.avg))
        if best_f1<g_mean:#클수록 좋단말이다!
            best_f1=g_mean
            nsml.save("best_fine")#always save best
            torch.save(model.state_dict(),'fine_best.pt')

        save_idx+=1

    

##result =0.0 제출파일형식에 문제가잇음을 말함
def evaluate(model,test_dir):
    ####test 를 위한 evaluate부분!!


    data_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_info = preprocess_test_info(test_dir)
    test_dataset = SpamDataset(test_info.img_path.values,
                                test_info.index.values,
                                tfms=data_transform,
                                test=True)
    test_dloader=DataLoader(test_dataset, batch_size=32, num_workers=4)

    
    model.eval()
    y_pred=[]
    files=[]
    for idx,data in enumerate(test_dloader):
        #monotone_list=[]
        img=data[0].cuda()
        file_name=data[1]#filename
        
        """
        if img.std()<0.05:#monotone
            monotone_list.append(idx)
        """
        #labels=data[1].cuda()
        out=model(img)
        output_prob = F.softmax(out, dim=1)
        _,pred=torch.max(output_prob,1)
        pred=pred.cpu().numpy().tolist()
        """
        for i in monotone_list:
            pred[i]=1 #수동으로 monotone에 넣어줌
        """  
        files+=file_name
        y_pred+=pred
    
    ret = pd.DataFrame({'filename': files, 'y_pred': y_pred})
    return ret

class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(hue=.05, saturation=.05),])
        self.strong = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=256,
                                  padding=int(256*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='parser')
    #nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    
    #custom
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()
    #model
    #model=Res152(4).cuda()
    model=EfficientNet.from_pretrained('efficientnet-b1',num_classes=4).cuda()
    #model=build_resnext(4,28,4,4).cuda()
    bind_model(model)
    #nsml.load(checkpoint='17', session='hyeinhyun/spam-3/61')#weight load

    if args.pause:
        nsml.paused(scope=locals())
        
    if args.mode == 'train':
    #dataset
        base_dir = Path(mkdtemp())
        label_transform = T.Compose([
        #SVHNPolicy(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(hue=.05, saturation=.05),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])#only resize
        val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_info, valid_info,unlabel_info = preprocess_train_info(base_dir, sup=False)
  
        train_dataset = SpamDataset(train_info.img_path.values,
                                    train_info.label.values,
                                    tfms=label_transform)
        valid_dataset = SpamDataset(valid_info.img_path.values,
                                    valid_info.label.values,
                                    tfms=val_transform)


        tr_dloader=DataLoader(train_dataset, batch_size=32, num_workers=4,shuffle=True)
        
        val_dloader=DataLoader(valid_dataset, batch_size=32, num_workers=4,shuffle=True)

        full_train2(tr_dloader,val_dloader,model,30) #full tuning

        print("full tune Fin\n")
        checkpoint = torch.load('full_best.pt')
        model.load_state_dict(checkpoint)
        #print("best fine tune model load Fin") 
        #print(len(valid_info.label.values))
        #sampler2=WeightedRandomSampler(valid_dataset.weight,1000,replacement=False)
 
        #full_train2(tr_dloader,val_dloader,model,40)
        #print("full tune Fin\n")
        #checkpoint = torch.load('full_best.pt')
        #model.load_state_dict(checkpoint)
        #print("best fine tune model load Fin") 
        #print(len(valid_info.label.values))
        #full_train2(tr_dloader,val_dloader,model,100)
        #tr_dloader=DataLoader(train_dataset, batch_size=8, num_workers=4,shuffle=True)
        #sampler1=WeightedRandomSampler(train_dataset.weight,10000,replacement=True)
        ul_dataset = ulDataset(unlabel_info.img_path.values,tfms=TransformFix(0.5,0.5))
        ul_dloader=DataLoader(ul_dataset, batch_size=64, num_workers=4,shuffle=True)
        tr_dloader=DataLoader(train_dataset, batch_size=8, num_workers=4,shuffle=True)
        #tr_dloader=DataLoader(train_dataset, batch_size=8, num_workers=4,sampler=sampler1)
        #val_dloader=DataLoader(valid_dataset, batch_size=32, num_workers=4,sampler=sampler2)


        fine_train(tr_dloader,val_dloader,ul_dloader,model,40)
        
        print("2fine tune Fin\n")
        checkpoint = torch.load('fine_best.pt')
        model.load_state_dict(checkpoint)
        print("best fine tune model load Fin")  


        #ul_dloader=DataLoader(ul_dataset, batch_size=32, num_workers=4,shuffle=True)#배치사이즈만 변경
        tr_dloader=DataLoader(train_dataset, batch_size=32, num_workers=4,shuffle=True)
        full_train(tr_dloader,val_dloader,model,50)



        #change train
        #sampler1=WeightedRandomSampler(train_dataset.weight,10000,replacement=False)


        #tr_dloader=DataLoader(train_dataset, batch_size=32, num_workers=4,sampler=sampler1)

    
    #model load
