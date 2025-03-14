""" Definition of supervised learning algorithms"""
import torch
import torch.nn as nn
import torchvision
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

class BaseNN(nn.Module):
    #TODO differentiate between num_classes and num_classes_epr_task
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.head_input_size = None

    def initialize_head(self):
        self.linear = nn.Linear(self.head_input_size, self.num_classes)

    def forward_head(self, features):
        return self.linear(features)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    def assign_weights(self, weight_vector):
        """ Changes the network weights to match the weight_vector """
        with torch.no_grad():
            start_idx = 0
            for param in self.parameters():
                param_length = param.numel()
                param.data = weight_vector[start_idx:start_idx + param_length].view(param.size())
                start_idx += param_length

class CustomCNN(BaseNN):
    def __init__(self, input_channels, num_classes, num_layers, base_channels=32, input_size=32):
        super(CustomCNN, self).__init__(num_classes)
        
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
        self.head_input_size = self._get_fc_input_size(input_channels)
        self.initialize_head()
        
        print(f"Number of parameters {self.count_parameters()/1e6} MLN")

    
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

    def _get_fc_input_size(self, input_channels):
        """ Compute the input size for the fully connected layer based on the input image size. """
        # Assuming the input image is 32x32 (common for CIFAR-like datasets)
        test_input = torch.zeros(1, input_channels, self.input_size, self.input_size)  # Example input tensor (batch_size, channels, height, width)
        output = self.conv_layers(test_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))  # Flatten output size

    def get_features(self, x):
        # Apply the convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the output of conv layers before feeding it into the fully connected layer
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        #Features 
        f = self.get_features(x)
        # Fully connected layer
        y = self.forward_head(f)
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(BaseNN):
    def __init__(self, block, num_blocks, num_classes=10, planes=64):
        """
        num_classes: output size FOR EACH HEAD (total output size = num_classes*num_heads)
        """
        super(ResNet, self).__init__(num_classes)
        self.in_planes = planes

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, planes*1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, planes*8, num_blocks[3], stride=2)
  
        self.head_input_size = self.in_planes
        self.initialize_head()
        
        print(f"Number of parameters {self.count_parameters()/1e6} MLN")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def get_features(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        """ 
        Returns a single value if single-head, otherwise, a list of outputs if t<0 or the output of t-th head. 
        """
        #Features 
        f = self.get_features(x)
        # Fully connected layer
        y = self.forward_head(f)
        return y


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def get_network_from_name(net_name, **kwargs):

    if net_name == "resnet18":
        return ResNet18(num_classes=kwargs.get("num_classes_total"))
    elif net_name == "resnet34":
        return ResNet34(num_classes=kwargs.get("num_classes_total"))
    elif net_name == "resnet50":
        return ResNet50(num_classes=kwargs.get("num_classes_total"))
    elif net_name == "CNN":
        return CustomCNN(input_channels=kwargs.get("input_channels",3), 
                         num_classes=kwargs.get("num_classes_total"), 
                         num_layers=kwargs.get("net_depth",8), 
                         base_channels=kwargs.get("base_channels",4), 
                         input_size=kwargs.get("input_size", 224))
    else:
        raise ValueError(f"Unknown network name: {net_name}")
