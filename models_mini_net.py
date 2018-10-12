## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
		
		# Covolutional Layers
		# W_out = ((ImageWidth-Kernelsize + 2xPadding)/Stride)+1 = ((224-16 + 2x0)/1)+1 = 209 connections
        self.conv_1 = nn.Conv2d(1, 16, 5) # in_channels = 1; out_channels = 16; kernel_size = 5x5, stride=(1, 1) default; padding = 0 default  
        self.conv_2 = nn.Conv2d(16, 64, 3) # in_channels = 16; out_channels = 64; kernel_size = 3x3, stride=(1, 1) default; padding = 0 default 
        self.conv_3 = nn.Conv2d(64, 256, 3) # in_channels = 64; out_channels = 256; kernel_size = 3x3, stride=(1, 1) default; padding = 0 default
        self.conv_4 = nn.Conv2d(256, 512, 1) # in_channels = 256; out_channels = 512; kernel_size = 1x1, stride=(1, 1) default; padding = 0 default 

		# Maxpooling Layer	(for all)	
        self.pool = nn.MaxPool2d(2, 2)
		
		# Dropout (for all)	
        self.drop = nn.Dropout(0.6)
		
		# Fully Connected Layers (fc)
        #self.fc_1 = nn.Linear(in_features = 73728, out_features = 5000)  
        #self.fc_2 = nn.Linear(in_features = 5000, out_features =  5000)  
        #self.fc_3 = nn.Linear(in_features =  5000, out_features =   136)    #68 keypoints with x and y coordinate => out_features: 136
		
        #self.fc_1 = nn.Linear(in_features = 86528, out_features = 1000)  
        #self.fc_2 = nn.Linear(in_features = 1000, out_features =  1000)  
        #self.fc_3 = nn.Linear(in_features = 1000, out_features =  1000)  
        #self.fc_4 = nn.Linear(in_features =  1000, out_features =   136)    #68 keypoints with x and y coordinate => out_features: 136
	
        self.fc_1 = nn.Linear(in_features = 173056, out_features = 1000)  
        self.fc_2 = nn.Linear(in_features = 1000, out_features =  1000)  
        self.fc_3 = nn.Linear(in_features =  1000, out_features =   136)    #68 keypoints with x and y coordinate => out_features: 136
		
		# Epoch: 3, Batch: 260, Avg. Loss: 0.0007566223330795765 (Kernelsize 5,3,3,1 and Dropout(0.6); 173056 in_features)
		# Avg. Loss: from 0.0040 - 0.0020
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
		# Covolutional Layer
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))

		# Flatten the layer
        x = x.view(x.size(0), -1)
		
        # print("in_features size: ", x.size(1))

        x = self.drop(F.relu(self.fc_1(x)))
        x = self.drop(F.relu(self.fc_2(x)))
        x = self.fc_3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
