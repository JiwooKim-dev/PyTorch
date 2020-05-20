#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader


# In[2]:


from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data_root = './Cellphone_data/training/'

'''
# 이미지 리사이즈
trans = transforms.Compost([
    transforms.Resize((64, 128))  # height, width
])
'''

train_data = torchvision.datasets.ImageFolder(root=data_root)

for num, value in enumerate(train_data):
    data, label = value
    
    if(label == 0):
        # resized된 이미지 저장
        # 폴더는 미리 만들어놔야 함. 자동으로 생성 안됨
        data.save('./Cellphone_data/training/label_0/%d_%d.jpeg'%(num, label))
    else:
        data.save('./Cellphone_data/training/label_1/%d_%d.jpeg'%(num, label))
    


# In[ ]:




