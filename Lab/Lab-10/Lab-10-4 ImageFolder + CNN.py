#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)


# In[3]:


trans = transforms.Compose([transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(root='./custom_data/train_data', transform=trans)


# In[4]:


data_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True, num_workers = 2)


# In[5]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16*13*29, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        return out


# In[6]:


# model testing
net = CNN().to(device)
test_input = (torch.Tensor(3,3,64,128)).to(device)
test_out = net(test_input)


# In[7]:


model = CNN().to(device)


# In[22]:


optimizer = optim.Adam(model.parameters(), lr = 0.00005)
loss_func = nn.CrossEntropyLoss().to(device)


# In[23]:


total_batch = len(data_loader)


# In[25]:


print('####### Learning Started #######\n\n')

epochs = 5
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        output = model(imgs)
        cost = loss_func(output, labels)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        avg_cost += cost.item() / total_batch
    
    print('Epoch {:2d}  Cost: {:.10f}'.format(epoch+1, avg_cost))

print('\n\n####### Learning Finished #######')


# In[35]:


''' Save Model '''
torch.save(model.state_dict(), "./model/model.pth")


# In[36]:


new_net = CNN().to(device)


# In[37]:


new_net.load_state_dict(torch.load('./model/model.pth'))


# In[38]:


print(model.layer1[0])
print(new_net.layer1[0])

print(model.layer1[0].weight[0][0][0])
print(new_net.layer1[0].weight[0][0][0])


# In[39]:


model.layer1[0].weight[0] == new_net.layer1[0].weight[0]


# In[40]:


trans = transforms.Compose([
    transforms.Resize((64,128)),
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(root='./custom_data/test_data', transform=trans)
test_set = DataLoader(dataset=test_data, batch_size=len(test_data))


# In[41]:


with torch.no_grad():
    for num, data in enumerate(test_set):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        
        prediction = new_net(imgs)
        correct_prediction = torch.argmax(prediction, 1) == label
        
        accuracy = correct_prediction.float().mean()
        
        print('Accuracy: ', accuracy.item())


# In[ ]:




