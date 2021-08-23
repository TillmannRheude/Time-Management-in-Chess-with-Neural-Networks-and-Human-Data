import torch.nn as nn

class CTM_Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output):
        super(CTM_Net, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden1_size)
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.linear3 = nn.Linear(hidden2_size, output)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.leaky_relu(x)

        x = self.linear2(x)
        x = nn.functional.leaky_relu(x)

        x = self.linear3(x)
        
        return x

class CTM_Net_v2(nn.Module):
    def __init__(self, output = 1):
        super(CTM_Net_v2, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels = 13, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1, padding = 2) 
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        # Fully connected layers
        self.fc4 = nn.Linear(in_features = 2048, out_features = output)
        
        # Dropout layer for better generalization
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv4(x)
        x = nn.functional.leaky_relu(x)
        
        x = x.view(-1, 2048)

        x = self.dropout(x)

        x = self.fc4(x)

        return x
 
class CTM_Net_v2_1(nn.Module):
    def __init__(self, output = 1):
        super(CTM_Net_v2_1, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels = 13, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 

        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.conv6 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 
        self.conv7 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 

        # Fully connected layers
        self.fc4 = nn.Linear(in_features = 2048, out_features = output)
        
        # Dropout layer for better generalization
        self.dropout = nn.Dropout(0.5) #0.15


    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)

        identity = x  
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv3(x)
        x += identity
        x = nn.functional.leaky_relu(x)

        identity = x  
        x = self.conv4(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv5(x)
        x += identity
        x = nn.functional.leaky_relu(x)

        identity = x  
        x = self.conv6(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv7(x)
        x += identity
        x = nn.functional.leaky_relu(x)
        
        x = x.view(-1, 2048)

        x = self.dropout(x)

        x = self.fc4(x)

        return x