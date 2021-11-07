from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
class CIFAR():
    def __init__(self, train_batch_size: int = 20, test_batch_size: int = 32):
        self.train_set = CIFAR10(root='./cifar10', train=True, download=True, transform=train_transform)
        self.test_set = CIFAR10(root='./cifar10', train=False, download=True, transform=test_transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=train_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=test_batch_size, shuffle=False)
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_dataset_size(self, mode: str='train'):
        if mode == 'train':
            return len(self.train_set)
        else:
            return len(self.test_set)
    
    def get_num_classes(self):
        return 10