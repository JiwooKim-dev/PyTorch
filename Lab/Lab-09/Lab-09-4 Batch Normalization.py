#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

batch_size = 100
training_epochs = 15
learning_rate = 0.001

linear1 = torch.nn.Linear(784, 32, bias=True)
linear2 = torch.nn.Linear(32, 32, bias=True)
linear3 = torch.nn.Linear(32, 10, bias=True)
relu = torch.nn.ReLU()
bn1 = torch.nn.BatchNorm1d(32)
bn2 = torch.nn.BatchNorm1d(32)

nn_linear1 = torch.nn.Linear(784, 32, bias=True)
nn_linear2 = torch.nn.Linear(32, 32, bias=True)
nn_linear3 = torch.nn.Linear(32, 10, bias=True)

bn_model = torch.nn.Sequential(linear1, bn1, relu, linear2, bn2, relu, linear3)  # Batch Norm 적용 model
nn_model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3)            # 미적용 model

loss = torch.nn.CrossEntropyLoss()
bn_optimizer = torch.optim.Adam(bn_model.parameters(), lr=learning_rate)
nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

mnist_train = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

torch.nn.init.xavier_uniform_(nn_linear1.weight)
torch.nn.init.xavier_uniform_(nn_linear2.weight)
torch.nn.init.xavier_uniform_(nn_linear3.weight)


for epoch in range(training_epochs):
    
    bn_model.train()
    
    bn_avg_cost = 0
    nn_avg_cost = 0
    total_batch = len(data_loader)
    
    for X, Y in data_loader:
    
        X = X.view(-1, 28*28)
        
        bn_hypothesis = bn_model(X)
        bn_cost = loss(bn_hypothesis, Y)
        bn_optimizer.zero_grad()
        bn_cost.backward()
        bn_optimizer.step()
        
        nn_hypothesis = nn_model(X)
        nn_cost = loss(nn_hypothesis, Y)
        nn_optimizer.zero_grad()
        nn_cost.backward()
        nn_optimizer.step()
        
        bn_avg_cost += bn_cost / total_batch
        nn_avg_cost += nn_cost / total_batch
    
    print('Epoch {:3d} [BN] Cost: {:.9f}   [NN] Cost: {:.9f}'.format(epoch, bn_avg_cost, nn_avg_cost))
    

'''
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28*28).float()
    Y_test = mnist_test.test_labels
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: {:.6f}'.format(accuracy.item()))
'''


# In[2]:


batch_size = 100
training_epochs = 15
learning_rate = 0.001

linear1 = torch.nn.Linear(784, 32, bias=True)
linear2 = torch.nn.Linear(32, 32, bias=True)
linear3 = torch.nn.Linear(32, 10, bias=True)
relu = torch.nn.ReLU()
bn1 = torch.nn.BatchNorm1d(32)
bn2 = torch.nn.BatchNorm1d(32)

nn_linear1 = torch.nn.Linear(784, 32, bias=True)
nn_linear2 = torch.nn.Linear(32, 32, bias=True)
nn_linear3 = torch.nn.Linear(32, 10, bias=True)

bn_model = torch.nn.Sequential(linear1, bn1, relu, linear2, bn2, relu, linear3)  # Batch Norm 적용 model
nn_model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3)            # 미적용 model

loss = torch.nn.CrossEntropyLoss()
bn_optimizer = torch.optim.Adam(bn_model.parameters(), lr=learning_rate)
nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

mnist_train = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

'''
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

torch.nn.init.xavier_uniform_(nn_linear1.weight)
torch.nn.init.xavier_uniform_(nn_linear2.weight)
torch.nn.init.xavier_uniform_(nn_linear3.weight)
'''


for epoch in range(training_epochs):
    
    bn_model.train()
    
    bn_avg_cost = 0
    nn_avg_cost = 0
    total_batch = len(data_loader)
    
    for X, Y in data_loader:
    
        X = X.view(-1, 28*28)
        
        bn_hypothesis = bn_model(X)
        bn_cost = loss(bn_hypothesis, Y)
        bn_optimizer.zero_grad()
        bn_cost.backward()
        bn_optimizer.step()
        
        nn_hypothesis = nn_model(X)
        nn_cost = loss(nn_hypothesis, Y)
        nn_optimizer.zero_grad()
        nn_cost.backward()
        nn_optimizer.step()
        
        bn_avg_cost += bn_cost / total_batch
        nn_avg_cost += nn_cost / total_batch
    
    print('Epoch {:3d} [BN] Cost: {:.9f}   [NN] Cost: {:.9f}'.format(epoch, bn_avg_cost, nn_avg_cost))
    


# In[ ]:




