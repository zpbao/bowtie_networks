import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models.resnet import model_urls
import torch.optim as optim

import numpy as np

model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')


class ResNet_C(nn.Module):
    def __init__(self, z_dim = 128, num_classes = 200):
        super(ResNet_C, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.z_dim)
        self.fc4 = nn.Linear(1000,256)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.tanh = nn.Tanh()
    
    def embed(self, x):
        h1 = self.fc1(x)
        h1 = self.relu(h1)
        h2 = self.fc2(h1)
        h2 = self.relu(h2)
        res = self.fc3(h2)
        res = self.tanh(res)
        return res
    
    def get_prototypes(self, f, Y):
        Yt = Y.t()
        sm = torch.mm(Yt, f)
        count = Yt.sum(1, keepdims = True)
        prototypes = sm / (count.expand_as(sm)+0.0001)
        return prototypes, count
    
    def get_scores(self, f, prototypes):
        prototypenorm = prototypes.pow(2).sum(1).t()
        dot = torch.mm(f, prototypes.t())
        scores = -prototypenorm.expand_as(dot) + 2*dot
        return scores

    def forward(self, x_source, x_target, Y_S):
        f1 = self.embed(x_source)
        f2 = self.embed(x_target)
        prototypes, count = self.get_prototypes(f1, Y_S)
        return self.get_scores(f2, prototypes)
