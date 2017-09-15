import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pdb
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##########################################
### Data loading:
##########################################
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,.5,.5), (.5,.5,.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + .5
    img = img.numpy()
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()

#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images))

##########################################
### Defining the Network:
##########################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()

##########################################
### Defining the loss and optimizer:
##########################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

##########################################
### Training the Network:
##########################################
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        
        # forward+backward+optimize:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if (i>0) and not (i%2000):
            print('[epoch_{}, iter_{}], loss:{}'.format(epoch+1, i+1, running_loss/2000.0))
            running_loss = 0.0

print('Training is over!')

##########################################
### Testing the Network:
##########################################
correct = 0.0
total = 0.0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    labels = labels.cuda()
    _, predictions = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predictions==labels).sum()

print('\nTest Accuracy: {}'.format(100.0*correct/total))
