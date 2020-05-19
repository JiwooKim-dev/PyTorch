#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# In[2]:


import visdom
vis = visdom.Visdom()


# In[3]:


vis.text('Hello, world!', env='main')  # env : 나중에 main만 종료 하면 한 번에 다 종료 가능


# In[4]:


sample_img = torch.randn(3, 200, 200)
vis.image(sample_img)


# In[5]:


vis.images(torch.Tensor(3,3,28,28))


# In[8]:


MNIST = dsets.MNIST(root='./MNIST_data', train=True, transform=transforms.ToTensor(), download=True)
cifar10 = dsets.CIFAR10(root='./MNIST_data', train=True, transform=transforms.ToTensor(), download=True)


# In[10]:


''' CIFAR10 예시 띄워보기 '''
data = cifar10.__getitem__(0)
print(data[0].shape)
vis.images(data[0], env='main')


# In[11]:


data_loader = torch.utils.data.DataLoader(dataset=MNIST,
                                         batch_size=32,
                                         shuffle=False)


# In[12]:


for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    break


# In[13]:


vis.close(env='main')  # 손으로 다 꺼도 됨


# In[21]:


''' Line Plot '''
Y_data = torch.randn(6)
plt = vis.line(Y=Y_data)


# In[22]:


X_data = torch.Tensor([1,2,3,4,5,6])
plt = vis.line(Y=Y_data, X=X_data)


# In[23]:


''' Line Update '''
Y_append = torch.randn(1)
X_append = torch.Tensor([7])

vis.line(Y=Y_append, X=X_append, win=plt, update='append')


# In[29]:


''' Multiple Lines in a window '''
num = torch.Tensor(list(range(0,10)))
print(num.shape)


# In[30]:


num = num.view(-1, 1)
print(num.shape)
num = torch.cat((num, num), dim=1)
print(num.shape)
plt = vis.line(Y=torch.randn(10,2), X=num)


# In[33]:


''' Line info '''
plt = vis.line(Y=Y_data, X=X_data, opts=dict(title='Test', showlegend=True))
plt = vis.line(Y=torch.randn(10,2), X=num, opts=dict(title='테스트', legend=['1번', '2번'], showlegend=True))


# In[34]:


''' Function for updating line '''
def loss_tracker(loss_plot, loss_val, num):
    vis.line(
            X=num,
            Y=loss_val,
            win=loss_plot,
            update='append'
            )


# In[36]:


plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))


# In[37]:


vis.close(env='main')


# In[ ]:




