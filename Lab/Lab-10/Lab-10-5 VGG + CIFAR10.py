#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models.vgg as vgg
import torchvision
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo


# In[2]:


import visdom

vis = visdom.Visdom()
vis.close(env='main')


# In[3]:


def loss_tracker(loss_plot, loss_value, num):
    vis.line(
        X=num,
        Y=loss_value,
        win=loss_plot,
        update='append'
        )


# In[4]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)


# In[5]:


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[6]:


dataiter = iter(trainloader)
images, labels = dataiter.next()
vis.images(images/2 + 0.5)


# In[7]:


cfg = [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M']  # vgg16


# In[8]:


class VGG(nn.Module):
    
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7,7))  이미지가 7x7 보다 작은 경우 안쓰는것이 좋음
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 4096),  # 32x32가 max pool 3번 거치면 4x4
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


# In[9]:


vgg16 = VGG(vgg.make_layers(cfg), 10, True).to(device)


# In[10]:


a = torch.Tensor(1, 3, 32, 32).to(device)
out = vgg16(a)
print(out)


# In[11]:


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(vgg16.parameters(), lr=0.005, momentum=0.9)

lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 점점 learning rate 줄여서 정교한 학습


# In[12]:


loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss tracker', legend=['loss'], showlegend=True))


# In[13]:


# training

epochs = 50

print('### Learning Started ###\n')

for epoch in range(epochs):
    running_loss = 0.0
    lr_sche.step()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 30 == 29:
            loss_tracker(loss_plt, torch.Tensor([running_loss/30]), torch.Tensor([i+epoch*len(trainloader)]))
            print('{:d} {:5d} loss: {:.3f}'.format(epoch+1, i+1, running_loss/30))

print('\n### Learning Finished ###')


# In[ ]:




