import numpy as np
import torch
import os
import codecs
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
#import gzip

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(0)

class FashionMNISTDataset(Dataset):
    processed_folder='processed'
    training_file='training.pt'
    test_file='test.pt'
    def __init__(self,root_dir,train=True,transform=transforms.ToTensor()):
        self.root=os.path.join(root_dir)
        self.transform=transform
        self.train=train
        
        if self.train:
            with open(os.path.join(self.root, 'train-images-idx3-ubyte'), 'rb') as f:
                data = f.read()
                length=int(codecs.encode(data[4:8],'hex'),16)
                num_rows = int(codecs.encode(data[8:12],'hex'),16)
                num_cols = int(codecs.encode(data[12:16],'hex'),16)
                parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                self.train_data = torch.from_numpy(parsed).view(length, num_rows, num_cols)

                
            with open(os.path.join(self.root, 'train-labels-idx1-ubyte'), 'rb') as f:
                data = f.read()
                parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                self.train_labels=torch.from_numpy(np.array([[int(parsed[i]==j) for j in range(10)]for i in range(len(parsed))]))
            
        else:
            with open(os.path.join(self.root, 't10k-images-idx3-ubyte'), 'rb') as f:
                data = f.read()
                length=int(codecs.encode(data[4:8],'hex'),16)
                num_rows = int(codecs.encode(data[8:12],'hex'),16)
                num_cols = int(codecs.encode(data[12:16],'hex'),16)
                parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                self.test_data = torch.from_numpy(parsed).view(length, num_rows, num_cols)
                
            with open(os.path.join(self.root, 't10k-labels-idx1-ubyte'), 'rb') as f:
                data = f.read()
                parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                self.test_labels=torch.from_numpy(np.array([[int(parsed[i]==j) for j in range(10)]for i in range(len(parsed))]))
                
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def __getitem__(self,idx):
        if self.train:
            img, target = self.train_data[idx], self.train_labels[idx]
        else:
            img, target = self.test_data[idx], self.test_labels[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        return img,target

train_dataset=FashionMNISTDataset(root_dir="./fashion-mnist/data/fashion/")
test_dataset=FashionMNISTDataset(root_dir="./fashion-mnist/data/fashion/",train=False)

batch_size=100
n_iters = 18000
num_epochs = (n_iters*batch_size)/len(train_dataset)
num_epochs=int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class Swish(nn.Module):
  def forward(self, input):
    return (input * torch.sigmoid(input))
  
  def __repr__(self):
    return self.__class__.__name__ + ' ()'

class CNNModel(nn.Module):
    def __init__ (self):
        super(CNNModel,self).__init__()
        
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.swish1=Swish()
        nn.init.xavier_normal(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=1)
        
        self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(32)
        self.swish2=Swish()
        nn.init.xavier_normal(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        
        self.cnn3=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.swish3=Swish()
        nn.init.xavier_normal(self.cnn3.weight)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*6*6,10)
        
        self.softmax=nn.Softmax(dim=1)
        
        
    def forward(self,x):
        out=self.cnn1(x)
        out=self.bn1(out)
        out=self.swish1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.bn2(out)
        out=self.swish2(out)
        out=self.maxpool2(out)
        out=self.cnn3(out)
        out=self.bn3(out)
        out=self.swish3(out)
        out=self.maxpool3(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        out=self.softmax(out)
        
        return out

model=CNNModel()

if torch.cuda.is_available():
    model.cuda()

criterion=nn.BCELoss()

learning_rate=0.015

optimizer=torch.optim.Adagrad(model.parameters(),lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)# this will decrease the learning rate by factor of 0.1 every 10 epochs

iter=0
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        model.train()
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs=model(images)
        
        loss=criterion(outputs.float(),labels.float())
        
        loss.backward()
        
        optimizer.step()
        iter+=1
        
        if iter%500 == 0:
            model.eval()
            correct=0
            total=0
            for images,labels in test_loader:
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
                labels=labels.float()
                outputs=model(images)
                predicted=outputs.data.float()
                _,predicted=torch.max(predicted,1)
                _,labels=torch.max(labels,1)
                total+=labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
                
            accuracy=100*float(correct)/total
            print('Iterations : {}, Loss : {}, Accuracy: {}'.format(iter,loss.data[0],accuracy))
    scheduler.step()
