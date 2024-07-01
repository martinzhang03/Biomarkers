import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Adding more convolutional and pooling layers to reduce the size gradually
        # self.c1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=2, bias=True)
        #self.c2 = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1)

        # self.conv_reduce = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Reduces 14x14 to 7x7
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)

        # model = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
        # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        # self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))
        # Adjust the input size for the first fully connected layer accordingly
        # self.fc1 = nn.Linear(64 * 50 * 50, 512)  # Adjust dimensions after pooling and convolution layers
        
        # self.simplier = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 1)  # Assuming 10 classes for the output

        # self.f1 = nn.Linear(512, 512)
        # self.f2 = nn.Linear(512, 1)
        # self.dropout = nn.Dropout(p=0.5)

        model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=24, stride = 2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * 3 * 3, 1)  # Adjusted input dimensions
        self.dropout1 = nn.Dropout(0.05)
        # self.fc2 = nn.Linear(128, 64)
        # self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(4, 1)

        # self.layer0 = nn.Sequential(
        #     nn.
        # )

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 16, stride=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        # )

        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, stride=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(stride=2, kernel_size=2),  # 30 80
        # )

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),   # 15 40
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(6272, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 40),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(40, 1)
        # )




        
    def forward(self, x):
        # x = self.c1(x) # (3 X 28 X 28) --> (3 X 14 X 14)
        # nn.ReLU()

        # # x = self.simplier(x) # (3 X 14 X 14) --> (512 X 14 X 14)
        # x = self.resnet(x) # (3 X 14 X 14) --> (512 X 1 X 1)
        # nn.ReLU()

        # # x = self.conv_reduce(x) # (512x14x14) --> (512x7x7)
        # # x = self.pool(x) # (512x7x7) --> (512x1x1)
        # # nn.ReLU()

        # x = torch.flatten(x, 1) # (512 X 1 X 1) --> (512)
        # x = self.f1(x)
        # nn.ReLU()

        # nn.Dropout(p=0.5, inplace=True) # (512) --> (512)

        # x = self.f2(x) # (512) --> (1)
        # return x

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = F.softmax(x, dim=-1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)

        # x = self.conv2(x)
        # nn.ReLU()
        
       # x = self.pool(x)

        # x = self.flatten(x)
        x = self.fc1(x)
        # x = self.dropout1(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        # x = self.output(x)

        return x

StartingNetwork = ConvNet

if (__name__ == "__main__"):
    print("Starting Network Debug...")
    # import torchsummary
    # new_network = StartingNetwork()
    # torchsummary.summary(new_network, (3, 400, 400))