from uti.get_Mnist import *
from torch.utils.data import TensorDataset, DataLoader

def draw_num(image_tensor):
    plt.imshow(image_tensor.numpy().reshape(28,28),cmap='gray')
    
def get_dl(train_ds,valid_ds,bs,**kwags):
    train_dl = DataLoader(
        dataset = train_ds,
        batch_size = bs,
        shuffle = True,
        **kwags
    )
    valid_dl = DataLoader(
        dataset = valid_ds,
        batch_size = bs * 2,
        **kwags
    )
    return train_dl,valid_dl

def get_dataset(x_train,y_train,x_valid=None,y_valid=None):
    if x_valid is None:
        return TensorDataset(x_train,y_train),None
    return TensorDataset(x_train,y_train),TensorDataset(x_valid,y_valid)

class Databunch():
    def __init__(self,train_dl,valid_dl,c=None):
        self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c
        
    @property
    def train_ds(self):
        return self.train_dl.dataset
    
    @property
    def valid_ds(self):
        return self.valid_dl.dataset