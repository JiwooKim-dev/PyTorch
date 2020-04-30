#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn.functional as F


# In[10]:


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 80],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
    
    def __len__(self):  # 데이터셋의 총 데이터 수
        return len(self.x_data)

    def __getitem__(self, idx):  # idx에 상응하는 입출력 데이터 리턴
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        return x, y
    
dataset = CustomDataset()


# In[8]:


from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size = 2,  # 각 minibatch의 크기 / 통상적으로 2의 제곱수 부여
    shuffle = True,  # 매 epoch마다 데이터 shuffle -> 학습 순서 변경
)


# In[12]:


import torch.nn as nn

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)  # 입력 3, 출력 1의 모델 생성
        
    def forward(self, x):
        return self.linear(x)  # hypotehsis 자동 계산
    
# Model
model = MultivariateLinearRegressionModel()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):  # minibatch 인덱스와 데이터
        x_train, y_train = samples
        
        # H(x)
        hypothesis = model(x_train)
        
        # Cost
        cost = F.mse_loss(hypothesis, y_train)
        
        # Gradient Descent
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:3d}/{} Batch {}/{} Cost: {:.5f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))
    


# In[ ]:




