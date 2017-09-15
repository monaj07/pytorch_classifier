
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import time
import copy
import os
import pdb

##########################################
### Data loading:
##########################################
data_transforms = {'train': transforms.Compose([
                            transforms.RandomSizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([.485,.456,.406], [.229,.224,.225])]),
                   'val':   transforms.Compose([
                            transforms.Scale(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([.485,.456,.406], [.229,.224,.225])])}

data_dir = 'hymenoptera_data'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dset_sizes = {'train': len(dsets['train']), 'val': len(dsets['val'])}
dset_classes = dsets['train'].classes


##########################################
### Training Function:
##########################################
def train_model(model, data_loaders, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()
    best_model = model
    best_Acc = 0.0

    for epoch in range(num_epochs):
        print('-'*20)
        LOSS = {'train':0.0, 'val':0.0}
        ACC = {'train':0.0, 'val':0.0}
        for phase in ['train','val']:
            if phase=='train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_correct = 0.0
            for data in data_loaders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    loss.backward()
                    optimizer.step()

                _, predictions = torch.max(outputs.data, 1)
                running_correct += torch.sum(predictions==labels.data)
                running_loss += loss.data[0]

            LOSS[phase] = running_loss/dset_sizes[phase]
            ACC[phase]  = running_correct/dset_sizes[phase]

        print('Epoch {0}: {1}_loss={2}, {1}_accuracy={3}'.format(epoch, 'train', LOSS['train'], ACC['train']))
        print('Epoch {0}: {1}_loss={2}, {1}_accuracy={3}'.format(epoch, 'val', LOSS['val'], ACC['val']))

        if (phase=='val') and best_Acc<ACC[phase]:
            best_model = copy.deepcopy(model)
            best_Acc = ACC[phase]
    time_elapsed = time.time() - since
    print('Training is finished in {} seconds, with the best validation accuracy of {}%'.format(time_elapsed, best_Acc))
    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch//lr_decay_epoch))
    if not(epoch % lr):
        print('Learning rate : {}.'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def imshow(img):
    inp = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)

def visualize_model(model, data_loaders, num_images=6):
    images_so_far = 0
    fig = plt.figure()
    for i, data in enumerate(data_loaders['val']):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[predictions[j][0]]))
            imshow(inputs.cpu().data[j])
            if images_so_far==num_images:
                return

model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.required_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

model_ft = train_model(model_ft, dset_loaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

visualize_model(model_ft, dset_loaders)
plt.show()
