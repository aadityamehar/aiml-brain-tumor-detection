import torch
import torch.nn as nn
import torch.nn.functional as F


class BTModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(3, 32, 3)
        self.conv3 = nn.Conv2d(3, 32, 3)
        self.conv4 = nn.Conv2d(3, 32, 3)
        
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.2)
        self.flatten = nn.Flatten()
        
        self.dense = nn.Linear(128)
        self.output = nn.Linear(1)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        output = F.sigmoid(self.output(x))
        
        return output
    