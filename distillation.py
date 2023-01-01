import torch
import torch.nn as nn

import torchvision.models as models
from torchvision.models.resnet import model_urls

model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')

import ResNetBasic

class Teacher(nn.Module):
    def __init__(self, num_classes = 200):
        super(Teacher, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet18(pretrained = True)
        self.classifier = nn.Sequential(nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.resnet(x)
        res = self.classifier(x)
        return res
    
    def fe(self, x):
        return self.resnet(x)

class Student(nn.Module):
    def __init__(self, num_classes = 200):
        super(Student, self).__init__()
        self.num_classes = num_classes
        self.resnet = ResNetBasic.ResNet18()
        self.classifier = nn.Sequential(nn.Linear(1000, num_classes))

    def forward(self, x):
        f = self.resnet(x)
        res = self.classifier(f)
        return f, res