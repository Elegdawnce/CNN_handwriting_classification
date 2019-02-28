import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

EPOCH = 20
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False
WEIGHT_DECAY = 0.0001
LR_DECAY_STEP_SIZE = 6


class MyDataset(Dataset):
    def __init__(self, documents,size):
        self.x = list()
        self.y = list()
        for doc in documents:
            self.x.append(doc[0])
            self.y.append(doc[1])
            size -= 1
            if size == 0 :
                break

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

# transform 将图片映射成（0-1）之间，原来是（0-255）
train_data = torchvision.datasets.MNIST(root = './mnist',
                                        train=True,
                                        transform =torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST)

test_data= torchvision.datasets.MNIST(root='./mnist',
                                      train=False,
                                      transform=torchvision.transforms.ToTensor())


train_dataset = MyDataset(train_data,8000)
test_dataset = MyDataset(test_data,2000)



# test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255
# test_y = test_data.test_labels[:2000]

# plot one example
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
# plt.title("%i" % train_data.train_labels[0])
# plt.show()

train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
test_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

