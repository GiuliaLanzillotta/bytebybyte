""" Definition of supervised learning algorithms"""
import torch
import torch.nn as nn
import torchvision
import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_layers, base_channels=32, input_size=32):
        super(CustomCNN, self).__init__()
        
        # Initialize layers dynamically based on num_layers and base_channels
        layers = []
        in_channels = input_channels
        self.input_size = input_size
        self.layer_input_size = input_size
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)  # Increase channels progressively
            layers.append(self._conv_block(in_channels, out_channels))
            in_channels = out_channels
        
        # Create a sequential model with convolutional layers
        self.conv_layers = nn.Sequential(*layers)
        print(self.conv_layers)
        
        # Compute the size of the input to the fully connected layer
        # Assuming input images are square (height == width)
        self.fc_input_size = self._get_fc_input_size(input_channels)
        
        # Fully connected layer
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        print(f"Number of parameters {self.count_parameters()/1e6} MLN")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    
    def _conv_block(self, in_channels, out_channels):
        """ Helper function to create a convolutional block with Conv2d, BatchNorm, ReLU, and MaxPool. """
        
        self.layer_input_size= (self.layer_input_size+1)//2 + 1 # pooling effect
        block= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([out_channels,self.layer_input_size,self.layer_input_size])
        )
        return block

    def _get_fc_input_size(self, input_channels, ):
        """ Compute the input size for the fully connected layer based on the input image size. """
        # Assuming the input image is 32x32 (common for CIFAR-like datasets)
        test_input = torch.zeros(1, input_channels, self.input_size, self.input_size)  # Example input tensor (batch_size, channels, height, width)
        output = self.conv_layers(test_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))  # Flatten output size

    def forward(self, x):
        # Apply the convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the output of conv layers before feeding it into the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        return x



def get_network_from_name(net_name, **kwargs):

    if net_name == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=kwargs.get("num_classes_per_task"))
    elif net_name == "resnet50":
        return torchvision.models.resnet50(weights=None, num_classes=kwargs.get("num_classes_per_task"))
    elif net_name == "CNN":
        return CustomCNN(input_channels=kwargs.get("input_channels",3), 
                         num_classes=kwargs.get("num_classes_per_task"), 
                         num_layers=kwargs.get("net_depth",8), 
                         base_channels=kwargs.get("base_channels",4), 
                         input_size=kwargs.get("input_size", 224))
    else:
        raise ValueError(f"Unknown network name: {net_name}")
    