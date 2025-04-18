#######################################################################################
# Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#######################################################################################

from __future__ import print_function, division
from transformers import AutoImageProcessor, ViTMAEForPreTraining

import torch
import torch.nn as nn
import torch.optim as optim
# from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
import random
import argparse
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils.cutout import Cutout

import timm

from utils.drs import random_one_ablation, gen_ablation_set_block, \
    gen_ablation_set_column_fix, gen_ablation_set_row_fix
NUM_CLASSES_DICT = {'imagenette':10,'imagenet':1000,'flowers102':102,'cifar':10,'cifar100':100,'svhn':10,'gtsrb':43, 'sun397':397}

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoint_drs_new',type=str)
# parser.add_argument("--data_dir",default='./../../../../public',type=str)
parser.add_argument("--data_dir",default='../',type=str)
# parser.add_argument("--data_dir",default='../data/',type=str)
# parser.add_argument("--data_dir",default='/public',type=str)
# parser.add_argument("--data_dir",default='/home/qlzhou4/data/',type=str)

parser.add_argument("--dataset",default='imagenet',type=str)
parser.add_argument("--model",default='vit_base_patch16_224',type=str)
parser.add_argument("--epoch",default=30,type=int)
parser.add_argument("--lr",default=-1,type=float)
parser.add_argument("--cutout_size",default=128,type=int)
parser.add_argument("--resume",action='store_true')
parser.add_argument("--n_holes",default=2,type=int)
parser.add_argument("--cutout",default=False,action='store_true')
parser.add_argument("--ablation_size",default=19,type=int)
parser.add_argument("--ablation_type",default="column",type=str)
print("3")
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"

MODEL_DIR=args.model_dir
# if not args.dataset=='imagenet':
DATA_DIR=os.path.join(args.data_dir,args.dataset)
print(DATA_DIR)
print(args)
# else:
#     DATA_DIR='/public/ziquanliu2/imagenet-1k'
#     print(DATA_DIR)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

n_holes = args.n_holes
cutout_size = args.cutout_size
ablation_size = args.ablation_size
ablation_type=args.ablation_type
dataset_name=args.dataset

if args.cutout:
    model_name = args.model + '_cutout{}_{}_{}.pth'.format(n_holes,cutout_size,args.dataset)
else:
    model_name = args.model + '_{}_drs_{}_{}_{}_{}.pth'.format(args.dataset,args.ablation_size,args.dataset,ablation_type,args.dataset)
print(model_name)
device = 'cuda'

if 'vit_base_patch16_224' in model_name:
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
elif 'resnetv2_50x1_bit_distilled' in model_name:
    model = timm.create_model('resnetv2_50x1_bit_distilled', pretrained=True)
elif 'resmlp_24_distilled_224' in model_name:
    model = timm.create_model('resmlp_24_distilled_224', pretrained=True)
elif 'mae_vit_base' in model_name:
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=1000)
    # model.reset_classifier(num_classes=NUM_CLASSES_DICT[dataset_name])
    # processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
    # model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')

# get data loader
if args.dataset in ['imagenette','imagenet']:
    config = resolve_data_config({}, model=model)
    ds_transforms = create_transform(**config)
    ds_transforms.transforms.append(torchvision.transforms.RandomHorizontalFlip())
    # ds_transforms.transforms.append(torchvision.transforms.ColorJitter())
    if args.cutout:
        ds_transforms.transforms.append(Cutout(n_holes=n_holes, length=cutout_size))
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,'train'),ds_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,'val'),ds_transforms)
    if args.dataset == 'imagenette':
        num_classes = 10

elif args.dataset in ['cifar','cifar100','svhn','gtsrb','flowers102']:
    config = resolve_data_config({'crop_pct':1}, model=model)###############################to decide
    ds_transforms = create_transform(**config)
    ds_transforms.transforms.append(torchvision.transforms.RandomHorizontalFlip())
    # ds_transforms.transforms.append(torchvision.transforms.ColorJitter())
    if args.cutout:
        ds_transforms.transforms.append(Cutout(n_holes=n_holes, length=cutout_size))
    if args.dataset == 'cifar':
        train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=ds_transforms)
        val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=ds_transforms)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=ds_transforms)
        val_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=ds_transforms)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(root=DATA_DIR, split='train', download=True, transform=ds_transforms)
        val_dataset = datasets.SVHN(root=DATA_DIR, split='test', download=True, transform=ds_transforms)
        num_classes = 10
    elif args.dataset == 'gtsrb':
        # ds_transforms.transforms.append(transforms.RandomHorizontalFlip())
        train_dataset = datasets.GTSRB(root=DATA_DIR, split='train', download=True, transform=ds_transforms)
        val_dataset = datasets.GTSRB(root=DATA_DIR, split='test', download=True, transform=ds_transforms)
        num_classes = 43
    elif args.dataset == 'flowers102':
        train_dataset = datasets.Flowers102(root=DATA_DIR, split='train', download=True, transform=ds_transforms)
        val_dataset = datasets.Flowers102(root=DATA_DIR, split='test', download=True, transform=ds_transforms)
        num_classes = 102
    elif args.dataset=='sun397':
        train_dataset = datasets.SUN397(root=DATA_DIR, split='train', download=True, transform=ds_transforms)
        val_dataset = datasets.SUN397(root=DATA_DIR, split='test', download=True, transform=ds_transforms)
        num_classes = 397

print(ds_transforms)


image_datasets = {'train':train_dataset,'val':val_dataset}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

if args.dataset=='imagenet':
    print(256)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256,shuffle=False,num_workers=4)

else:
    print(128)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,shuffle=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128,shuffle=False,num_workers=4)

dataloaders={'train':train_loader,'val':val_loader}
if ablation_type=="column":
    ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_column_fix(ablation_size)
elif ablation_type=="row":
    ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_row_fix(ablation_size)
# elif ablation_type=="block":
#     ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_block(ablation_size)

print('device:',device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=20 ,mask=False):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    ablation=random_one_ablation(ablation_list)
                    # plt.imshow(torch.where(ablation, inputs, torch.tensor(0.).cuda()).cpu()[0].permute(1,2,0))
                    # plt.imshow(inputs.cpu()[0].permute(1,2,0))
                    # plt.show()
                    outputs = model(torch.where(ablation, inputs, torch.tensor(0.).cuda()))
                    if isinstance(outputs,tuple):
                        outputs = (outputs[0]+outputs[1])/2
                        #outputs = outputs[0]

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and model_name and not args.dataset=='imagenet':
                scheduler.step()
                print(scheduler.get_last_lr())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('saving...')
                torch.save({
                    'epoch': epoch,
                    'state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()
                    }, os.path.join(MODEL_DIR,model_name))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if args.dataset!='imagenet':
    model.reset_classifier(num_classes=num_classes)
    print("rest num_classes as "+str(num_classes))
model = torch.nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

if args.dataset=='imagenet':
    lr=0.001
    weight=0.0001
    print(lr)
    print(weight)

# elif args.dataset=='cifar' or args.dataset=='cifar100':
else:
    lr=0.01
    # lr=0.1
    weight=0.0005
    print(lr)
    print(weight)

# else:
#     lr=args.lr
#     print(lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=weight)
if args.dataset=='imagenet':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)
    print("no change on lr")
else:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # exp_lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10, verbose=True)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.1)




#https://pytorch.org/tutorials/beginner/saving_loading_models.html
if args.resume:
    print('restoring model from checkpoint...')
    checkpoint = torch.load(os.path.join(MODEL_DIR,model_name))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    #https://discuss.pytorch.org/t/code-that-loads-sgd-warning_notcerts-to-load-adam-state-to-gpu/61783/3
    optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


model = train_model(model, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=args.epoch)

