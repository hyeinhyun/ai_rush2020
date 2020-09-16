import argparse
import os

import nsml
import torch
from nsml import DATASET_PATH
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler
import pandas as pd

import nsml_utils as nu
from configuration.config import logger, train_transform, test_transform
from data_loader import TagImageDataset
from utils import select_optimizer, select_model, evaluate, train
from sklearn.model_selection import train_test_split

#install
from efficientnet_pytorch import EfficientNet
from utils import LabelSmoothingCrossEntropy, FocalLoss
#custom
from augmentation import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from sklearn.model_selection import KFold

#custom model
from models.model import Rex100, InceptionV3
from pretrainedmodels import nasnetalarge,pnasnet5large,inceptionresnetv2,xception, se_resnet152,polynet

def train_process(args, model, train_loader, test_loader, optimizer, criterion, device):
    best_acc = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        train_loss, train_acc = train(model=model, train_loader=train_loader, optimizer=optimizer,
                                      criterion=criterion, device=device, epoch=epoch, total_epochs=args.num_epoch)
        model.eval()
        test_loss, test_acc, test_f1 = evaluate(model=model, test_loader=test_loader, device=device, criterion=criterion)

        report_dict = dict()
        report_dict["train__loss"] = train_loss
        report_dict["train__acc"] = train_acc
        report_dict["test__loss"] = test_loss
        report_dict["test__acc"] = test_acc
        report_dict["test__f1"] = test_f1
        report_dict["train__lr"] = optimizer.param_groups[0]['lr']
        nsml.report(False, step=epoch, **report_dict)
        if best_acc < test_acc:
            checkpoint = 'best'
            logger.info(f'[{epoch}] Find the best model! Change the best model.')
            nsml.save(checkpoint)
            best_acc = test_acc
        if (epoch + 1) % 1 == 0:
            checkpoint = f'ckpt_{epoch + 1}'
            nsml.save(checkpoint)

        if (epoch + 1) % args.annealing_period == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            logger.info('Learning rate annealed to : {lr:.6f} @epoch{epoch}'.format(
                epoch=epoch, lr=optimizer.param_groups[0]['lr']))


def load_weight(model, weight_file):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file).state_dict(), strict=True)
        print('load weight from {}.'.format(weight_file))
    else:
        print('weight file {} is not exist.'.format(weight_file))
        print('=> random initialized model will be used.')


def main():
    # Argument Settings
    parser = argparse.ArgumentParser(description='Image Tagging Classification from Naver Shopping Reviews')
    parser.add_argument('--sess_name', default='', type=str, help='Session name that is loaded')
    parser.add_argument('--checkpoint', default='best', type=str, help='Checkpoint')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_workers', default=10, type=int, help='The number of workers')
    parser.add_argument('--num_epoch', default=3000, type=int, help='The number of epochs')
    parser.add_argument('--model_name', default='resnet50', type=str, help='[resnet50, rexnet, dnet1244, dnet1222]')
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--learning_anneal', default=1.1, type=float)
    parser.add_argument('--annealing_period', default=10, type=int)
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--mode', default='train', help='Mode')
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)
    parser.add_argument('--weight_file', default='model.pth', type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    logger.info('Build Model')
    #model = InceptionV3(5)
    # model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    # model.last_linear = nn.Sequential(
    #     nn.Dropout(p=0.4),
    #     nn.Linear(in_features=1536, out_features=5, bias=True)
    # )
    #model = select_model(args.model_name, pretrain=args.pretrain, n_class=5)
    model = EfficientNet.from_pretrained('efficientnet-b3',num_classes=5).cuda()
    #model = pnasnet5large(num_classes=5, pretrained='imagenet')
    # model = pnasnet5large(num_classes=1000, pretrained='imagenet')
    # model.last_linear = nn.Sequential(
    #     nn.Dropout(p=0.4),
    #     nn.Linear(in_features=4320, out_features=5, bias=True)
    # )
    # model = xception(num_classes=1000, pretrained='imagenet')
    # model.last_linear = nn.Sequential(
    #     nn.Dropout(p=0.4),
    #     nn.Linear(in_features=2048, out_features=5, bias=True)
    # )
    # model = se_resnet152(num_classes=1000, pretrained='imagenet')
    # model.last_linear = nn.Sequential(
    #     nn.Dropout(p=0.4),
    #     nn.Linear(in_features=2048, out_features=5, bias=True)
    # )


    
    total_param = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model size: {total_param} tensors')
    #load_weight(model, args.weight_file)
    model = model.cuda()

    nu.bind_model(model)

    if args.pause:
        nsml.paused(scope=locals())

    if args.num_epoch == 0:
        nsml.save('best')
        return

    # Set the dataset
    logger.info('Set the dataset')
    df = pd.read_csv(f'{DATASET_PATH}/train/train_label')
    #df_trn,df_val = train_test_split(df, test_size=0.2,random_state=24)
    kf = KFold(n_splits=3)
    kfold = kf.split(df)
    next(kfold)
    next(kfold)
    trn_idx,val_idx=next(kfold)
    df_trn=df.iloc[trn_idx]
    df_val=df.iloc[val_idx]


    trainset = TagImageDataset(data_frame=df_trn, root_dir=f'{DATASET_PATH}/train/train_data',
                               transform=train_transform)
    testset = TagImageDataset(data_frame=df_val, root_dir=f'{DATASET_PATH}/train/train_data',
                              transform=test_transform)
    #sampler1=WeightedRandomSampler(trainset.weight,10000,replacement=True)
    #train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, num_workers=args.num_workers,sampler=sampler1)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    nSamples = [33661, 25218, 1564, 12152, 104157]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).cuda()
    #criterion = nn.CrossEntropyLoss(reduction='mean')
    # nSamples = [33661, 25218, 1564, 12152, 104157]
    # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    # normedWeights = torch.FloatTensor(normedWeights).cuda()
    #criterion = nn.CrossEntropyLoss(weight=normedWeights)
    #criterion = FocalLoss(gamma=0.1)

    #
    criterion = LabelSmoothingCrossEntropy()

    optimizer = select_optimizer(model.parameters(), args.optimizer, args.lr, args.weight_decay)

    criterion = criterion.cuda()

    if args.mode == 'train':


        logger.info('Start t train!')
        train_process(args=args, model=model, train_loader=train_loader, test_loader=test_loader,
                      optimizer=optimizer, criterion=criterion, device=device)

    elif args.mode == 'test':
        nsml.load(args.checkpoint, session=args.sess_name)
        logger.info('[NSML] Model loaded from {}'.format(args.checkpoint))

        model.eval()
        logger.info('Start to test!')
        test_loss, test_acc, test_f1 = evaluate(model=model, test_loader=test_loader, device=device,
                                                criterion=criterion)
        logger.info(test_loss, test_acc, test_f1)


if __name__ == '__main__':
    main()
