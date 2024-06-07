# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # # https://github.com/uclaacmai/leaf-us-alone

# # import itertools

# # class StartingNetwork(nn.Module):
# #   # Input Shape is a tuple of size 3 ex: (1, 224, 224)
# #   def __init__(self, input_shape=(1,224,224)):
# #     # Call nn.Module's constructor--don't forget this
# #     super().__init__()

# #     # Define layers
# #     # For our very simple model, we just flatten the inputs into a 1D tensor
# #     size = list(itertools.accumulate(input_shape, lambda x, y: x * y))[-1]
# #     print(size)

# #     self.flatten = nn.Flatten()
# #     self.fc1 = nn.Linear(size, 256)
# #     self.fc2 = nn.Linear(256, 128)
# #     self.fc3 = nn.Linear(128, 10)

# #   def forward(self, x):
# #     # Forward propagation
# #     x = self.flatten(x)
# #     x = self.fc1(x)
# #     x = F.relu(x)

# #     x = self.fc2(x)
# #     x = F.relu(x)

# #     x = self.fc3(x)

# #     # No activation function at the end
# #     # nn.CrossEntropyLoss takes care of it for us

# #     return x



# # class ConvNet(torch.nn.Module):
# #     """
# #     Basic logistic regression on 800x600x3 images.
# #     """

# #     def __init__(self):
# #         super().__init__()

# #         # TODO: Change the dimensions of layers to match that of our dataset (inputted)



# #         # Conv2d expects the following arguments
# #         #   - C, the number of channels in the input
# #         #   - C', the number of channels in the output
# #         #   - The filter size (called a kernel size in the documentation)
# #         #     Below, we specify 5, so our filters will be of size 5x5.
# #         #   - The amount of padding (default = 0)
# #         self.conv1 = nn.Conv2d(3, 4, kernel_size=5, padding=2) # notice how we use padding to prevent dimension reduction
# #         self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
# #         # Pooling layer takes two arguments
# #         #   - Filter size (in this case, 2x2)
# #         #   - Stride
# #         self.pool = nn.MaxPool2d(2, 2)

# #         self.fc1 = nn.Linear(160000, 256)
# #         self.fc2 = nn.Linear(256, 128)
# #         self.fc3 = nn.Linear(128, 5)

# #         # Original Code Here
# #         # self.flatten = nn.Flatten()
# #         # self.fc = nn.Linear(224 * 224 * 3, 1)
# #         # self.sigmoid = nn.Sigmoid()

# #     def forward(self, x):
# #         # Comments below give the shape of x
# #         # n is batch size

# #         # (n, 1, 28, 28)
# #         x = self.conv1(x)
# #         x = F.relu(x)
# #         # (n, 4, 28, 28)
# #         x = self.pool(x)
# #         # (n, 4, 14, 14)
# #         # x = self.conv2(x)
# #         # x = F.relu(x)
# #         # (n, 8, 14, 14)
# #         # x = self.pool(x)
# #         # (n, 8, 7, 7)
# #         x = torch.flatten(x, 1)
# #         # x = torch.reshape(x, (-1, 8 * 7 * 7))
# #         # (n, 8 * 7 * 7)
# #         x = self.fc1(x)
# #         x = F.relu(x)
# #         # (n, 256)
# #         x = self.fc2(x)
# #         x = F.relu(x)
# #         # (n, 128)
# #         x = self.fc3(x)
# #         # (n, 10)
# #         return x

# # StartingNetwork = ConvNet
# # if (__name__ == "__main__"):
# #   print("Starting Network Debug...")

# #   import torchsummary

# #   new_network = StartingNetwork()
# #   torchsummary.summary(new_network, (3, 400, 400))



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import itertools

# # class StartingNetwork(nn.Module):
# #   # Input Shape is a tuple of size 3 ex: (1, 224, 224)
# #   def __init__(self, input_shape=(1,224,224)):
# #     # Call nn.Module's constructor--don't forget this
# #     super().__init__()

# #     # Define layers
# #     # For our very simple model, we just flatten the inputs into a 1D tensor
# #     size = list(itertools.accumulate(input_shape, lambda x, y: x * y))[-1]
# #     print(size)

# #     self.flatten = nn.Flatten()
# #     self.fc1 = nn.Linear(size, 256)
# #     self.fc2 = nn.Linear(256, 128)
# #     self.fc3 = nn.Linear(128, 10)

# #   def forward(self, x):
# #     # Forward propagation
# #     x = self.flatten(x)
# #     x = self.fc1(x)
# #     x = F.relu(x)

# #     x = self.fc2(x)
# #     x = F.relu(x)

# #     x = self.fc3(x)

# #     # No activation function at the end
# #     # nn.CrossEntropyLoss takes care of it for us

# #     return x

# class ConvNet(nn.Module):
#     """
#     Basic logistic regression on 800x600x3 images.
#     """

#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(3, 4, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(8 * 100 * 100, 256)  # Adjusting for the pooling layer
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 107)  # 107 classes for ages -1 to 105

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x

# # Make sure to define the class name you want to import in the __init__.py
# StartingNetwork = ConvNet


import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/uclaacmai/leaf-us-alone

import itertools

# class StartingNetwork(nn.Module):
#   # Input Shape is a tuple of size 3 ex: (1, 224, 224)
#   def __init__(self, input_shape=(1,224,224)):
#     # Call nn.Module's constructor--don't forget this
#     super().__init__()

#     # Define layers
#     # For our very simple model, we just flatten the inputs into a 1D tensor
#     size = list(itertools.accumulate(input_shape, lambda x, y: x * y))[-1]
#     print(size)

#     self.flatten = nn.Flatten()
#     self.fc1 = nn.Linear(size, 256)
#     self.fc2 = nn.Linear(256, 128)
#     self.fc3 = nn.Linear(128, 10)

#   def forward(self, x):
#     # Forward propagation
#     x = self.flatten(x)
#     x = self.fc1(x)
#     x = F.relu(x)

#     x = self.fc2(x)
#     x = F.relu(x)

#     x = self.fc3(x)

#     # No activation function at the end
#     # nn.CrossEntropyLoss takes care of it for us

#     return x



class ConvNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5, padding=2) # notice how we use padding to prevent dimension reduction
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(160000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 107)

        # Original Code Here
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(224 * 224 * 3, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # (n, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        # (n, 4, 28, 28)
        x = self.pool(x)
        # (n, 4, 14, 14)
        # x = self.conv2(x)
        # x = F.relu(x)
        # (n, 8, 14, 14)
        # x = self.pool(x)
        # (n, 8, 7, 7)
        x = torch.flatten(x, 1)
        # x = torch.reshape(x, (-1, 8 * 7 * 7))
        # (n, 8 * 7 * 7)
        x = self.fc1(x)
        x = F.relu(x)
        # (n, 256)
        x = self.fc2(x)
        x = F.relu(x)
        # (n, 128)
        x = self.fc3(x)
        # (n, 10)
        return x

StartingNetwork = ConvNet
if (__name__ == "__main__"):
  print("Starting Network Debug...")

  import torchsummary

  new_network = StartingNetwork()
  torchsummary.summary(new_network, (3, 400, 400))

