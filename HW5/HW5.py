import os
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans

from torch.utils import data
from sklearn.metrics import accuracy_score
from resnet import ResNet50
from densenet import DenseNet121


# ## Download dataset 
# https://drive.google.com/drive/u/3/folders/1sHh6NvuKX6RB5OytLwf4kaqfQ9svJNDQ

# ## Load data

# In[3]:


x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[4]:


# It's a multi-class classification problem 
class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
               'dog': 5, 'frog': 6,'horse': 7,'ship': 8, 'truck': 9}
print(np.unique(y_train))


# ![image](https://img-blog.csdnimg.cn/20190623084800880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lqcDE5ODcxMDEz,size_16,color_FFFFFF,t_70)

# ## Data preprocess

# In[46]:


class MyCIFARDataset(data.Dataset):
    def __init__(self, images, labels=None, transform=None):
        if not (labels is None):
            assert len(images) == len(labels)
        
        self.images = images
        if transform is None:
            self.trans = trans.ToTensor()
        else:
            self.trans = transform
        
        self.labels = labels
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        img = self.trans(img)
        if not (self.labels is None):
            return img, self.labels[index][0]
        else:
            return img


# ## Build model & training

# In[57]:


class MyModel:
    def __init__(self, net):
        self.net = net.cuda()

        # Initiate SGD optimizer
        self.opt = torch.optim.SGD(
            net.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        #self.lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.opt,
        #    factor=0.1,
        #    patience=50,
        #    verbose=True
        #)
        self.lr_sch = torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones=[150, 250, 350], 
            gamma=0.1
        )


        # Use cross-entropy as loss function
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.train_trans = trans.Compose([
                trans.ToPILImage(),
                trans.RandomCrop(32, padding=4),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(degrees=15),
                trans.ToTensor(),
                trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.test_trans = trans.Compose([
                trans.ToPILImage(),
                trans.ToTensor(),
                trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


    def fit(
        self, x_train, y_train,
        batch_size=16,
        epochs=100,
        validation_data=None,
        shuffle=True,
        store_model=True
    ):
        # Data preprocessing
        self.train_set = MyCIFARDataset(x_train, y_train, self.train_trans)
        self.train_loader = data.DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=4
        )

        self.use_val = not (validation_data is None)
        if self.use_val:
            self.val_set = MyCIFARDataset(validation_data[0], validation_data[1], self.test_trans)
            self.val_loader = data.DataLoader(
                dataset=self.val_set,
                batch_size=16,
                shuffle=False,
                drop_last=False,
                num_workers=4
            )
        
        best_acc = 0.
        # Start training
        for ep in range(epochs):
            self.net.train()
            # Train
            avg_loss = 0.
            for input, label in self.train_loader:
                input, label = input.cuda(), label.cuda()
                output = self.net(input)
                loss = self.criterion(output, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                avg_loss += loss.item()
                
            avg_loss /= len(self.train_loader)
            
            #self.lr_sch.step(avg_loss)
            self.lr_sch.step()
            print('Epoch {}/{} ; loss={:.4f}'.format(ep, epochs, loss.item()))
            
            if self.use_val:
                self.net.eval()
                # Evaluate
                correct = 0.
                total = 0.
                for input, label in self.val_loader:
                    input, label = input.cuda(), label.cuda()
                    output = self.net(input)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(label.data).cpu().sum()
                    total += len(label.data)
                acc = (correct / total).item()
                print('Accuracy = ', acc)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.net.state_dict(), 'Best_model.pth')

    def predict(self, x_test, batch_size=16):
        self.test_set = MyCIFARDataset(x_test, transform=self.test_trans)
        self.test_loader = data.DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        preds = np.array([])
        for input in self.test_loader:
            input = input.cuda()
            output = self.net(input)
            pred = output.data.max(1)[1].cpu().numpy()
            preds = np.concatenate((preds, pred))
        return preds


# In[58]:


# Builde model
# Load model made by myself
#net = ResNet50(num_classes=len(class_index))

# Pretrained model
#net = torchvision.models.densenet121(pretrained=True)
net = DenseNet121()
#net.classifier = nn.Sequential(
#    nn.Dropout(0.5),
#    nn.Linear(1024, 512, True),
#    nn.BatchNorm1d(512),
#    nn.ReLU(),
#    nn.Dropout(0.5),
#    nn.Linear(512, 256, True),
#    nn.BatchNorm1d(256),
#    nn.ReLU(),
#    nn.Linear(256, len(class_index))
#)

model = MyModel(net)


# In[59]:


# Setup some hyperparameters
batch_size = 128
epochs = 600

# Fit the data into model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)


# ## DO NOT MODIFY CODE BELOW!
# **Please screen shot your results and post it on your report**

# In[ ]:


y_pred = your_model.predict(x_test)


# In[ ]:


assert y_pred.shape == (10000,)


# In[ ]:


y_test = np.load("y_test.npy")
print("Accuracy of my model on test set: ", accuracy_score(y_test, y_pred))


# In[ ]:




