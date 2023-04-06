import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import timm.optim.optim_factory as optim_factory
from trainer import trainer, validater
from model import SimpleCNN
import os
import torchvision.models as models

import time

if torch.cuda.is_available():
     device = 'cuda:0'
else:
     device = 'cpu'
def main(args):
    # model and device
    #model = SimpleCNN(args.num_class)
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.heads = torch.nn.Linear(768, 100)
    model.to(device)
    model = model.to(device)
    # dataset
    transform = T.Compose([
        T.Resize(256),  # (256, 256) で切り抜く。
        T.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く

        # Data Aug
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

        T.ToTensor(),  # テンソルにする。
        T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 標準化する。
        ])
    train_dataset = datasets.CIFAR100(
             './data',
             train  = True,
             download = True,
             transform=transform
        )
    train_size = int(len(train_dataset) * 0.9)
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batchsize,
        shuffle = True
    )
    valid_dataloader = DataLoader(
        dataset = valid_dataset,  
        batch_size = args.batchsize
    )
    # optimizer
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.1)

    # loss
    criterion = nn.CrossEntropyLoss()
    # Training
    start = time.time()
    early_stopping = [np.inf, 15, 0]
    for epoch in range(args.epochs):
        print(f'-------- epoch {epoch} --------')
        # train
        train_loss = trainer(train_dataloader, model, device, optimizer, epoch, criterion, args)
        # validate
        with torch.no_grad():
            valid_loss = validater(valid_dataloader, model, device, criterion)
        # early stopping
        if valid_loss < early_stopping[0]:
            early_stopping[0] = valid_loss
            early_stopping[-1] = 0
            p = args.model_dir
            if not os.path.isdir(p):
                os.makedirs(p)
            print ("valid_loss < early_stopping[0]")
            torch.save(model.state_dict(), os.path.join(p, str(epoch)+'.pt'))
        else:
            print ("ELSE")
            early_stopping[-1] += 1
            if early_stopping[-1] == early_stopping[1]:
                break
    end = time.time() - start
    f = open('time.txt', 'a')
    f.write("\nvit_aug_trans\n")
    f.write(str(end))
    f.close()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--batchsize', default=64)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--min_lr', default=1e-8)
    parser.add_argument('--warmup', default=10)
    parser.add_argument('--num_class', default=100)
    parser.add_argument('--model_dir', default='./models/C100/vit_b_16_aug_trans')
    args = parser.parse_args()
    main(args)