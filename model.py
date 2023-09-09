import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as func
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms




class CNN(nn.Moudle):
    def __init__(self, in_channels = 3, num_classes = 10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), stride=(1,1) padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1) padding(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
num_classes = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset=train.dataset, batch_size= batch_size, shuffle = True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset=train.dataset, batch_size= batch_size, shuffle = True)

#initialize Model 
model = CNN(input_size=input_size, num_classes=num_classes).todevice()

# Loss & Optimizer

criterion = nn.CrossEntropyLoss()
Optimizer = optim.Adam(modle.parameters(), lr = learning_rate)

# Train network

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader)

        data = data.to(device=device)
        targets = targets.to(device=device)
        
        data = data.reshape(data.shape[0], -1)
        
        scores = model(data)
        loss = criterion(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        