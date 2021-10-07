import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
class LeNet(Module):
    def __init__(self, numChannels, classes, args):
        # call the parent constructor
        super(LeNet, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
        kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
        kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=200, out_features=100)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=100, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

        self.lossFn = nn.NLLLoss()
        # set the device we will be using to train the model
        # self.device = torch.device("cpu")
        self.device = args.device
        
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

    def trainer(self, train_library, optimizer):
        #train lib : (nx3x40x40, n)
        X = np.squeeze(np.asarray(train_library[0]), axis=1)
        Y = np.asarray(train_library[1])
        Y_c = np.zeros((len(Y), 6))
        for i, y in enumerate(Y):
            Y_c[i][y-1] = 1
        
        # send the input to the device
        X = torch.FloatTensor(X)
        Y_c = torch.FloatTensor(Y_c)#.float()
        X, Y_c = X.to(self.device), Y_c.to(self.device)

        meanTrainLoss = 0
        meanCorrect = 0
        for _ in range(100):
            totalTrainLoss = 0
            trainCorrect = 0

            pred = self.forward(X.float())
            loss = self.lossFn(pred, Y_c.argmax(1))
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss = loss
            meanTrainLoss += loss
            
            trainCorrect = (pred.cpu().argmax(1) == Y_c.cpu().argmax(1)).type(torch.float).sum().item()
            meanCorrect += trainCorrect

        # return  totalTrainLoss.item(), trainCorrect
        return meanTrainLoss/100, meanCorrect/100

    def predict(self, test_library):
        with torch.no_grad():
            X = np.asarray(test_library[0])
            Y = np.asarray(test_library[1])
            Y_c = np.zeros((len(Y), 6))
            for i, y in enumerate(Y):
                Y_c[i][y-1] = 1

            X = torch.from_numpy(X)
            Y_c = torch.from_numpy(Y_c)
            X, Y_c = X.to(self.device), Y_c.to(self.device)
            pred = self.forward(X.float())
            testCorrect = (np.array(pred.cpu()).argmax(1) == np.array(Y_c.cpu().argmax(1))).sum().item()
        return  testCorrect
    
    def predict_label(self, single_map):
        # print("predict map[2]", single_map[:,2,:,:])
        with torch.no_grad():
            X= single_map.to(self.device)
            pred = self.forward(X.float())
        return  pred.argmax(1)