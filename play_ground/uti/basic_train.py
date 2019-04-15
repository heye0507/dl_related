from uti.datablock import *
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

def get_model(data,lr=0.5,nh=50):
    m = data.train_ds.tensors[0].shape[1]
    model = nn.Sequential(nn.Linear(m,nh),nn.ReLU(),nn.Linear(nh,data.c))
    opt = optim.SGD(model.parameters(),lr=lr)
    return model,opt

class Learner():
    def __init__(self, model,opt,loss_func,data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data