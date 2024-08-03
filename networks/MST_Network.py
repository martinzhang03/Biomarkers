import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1 - First Neural Network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=28, stride=1, padding=0)
        # self.dropout1 = nn.Dropout(0.05)
        # self.fc1 = nn.Linear(1 * 3 * 3, 1)  # Adjusted input dimensions
        # self.fc1 = nn.Linear(3*3, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        # x = self.dropout1(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        return x

class MultiStageNet(nn.Module):
    def __init__(self):
        super(MultiStageNet, self).__init__()
        # Stage 1: Five ConvNet instances
        self.stage1_nets = nn.ModuleList([ConvNet() for _ in range(5)])
        
        # Stage 2: One fully connected network
        self.stage2_fc1 = nn.Linear(5, 1)
        self.stage2_dropout = nn.Dropout(0.05)
        self.stage2_fc2 = nn.Linear(8, 1)

    def forward(self, x, category_indices):
        batch_size = x.size(0)
        # stage1_outputs = torch.zeros(batch_size, 5*9).to(x.device)
        stage1_outputs = torch.zeros(batch_size, 5).to(x.device)
        
        # for i in range(batch_size):
        #     category_idx = category_indices[i]
        #     stage1_output = self.stage1_nets[category_idx](x[i].unsqueeze(0))
        #     stage1_outputs[i, category_idx * 9:(category_idx + 1) * 9] = stage1_output
        for i in range(batch_size):
            category_idx = category_indices[i]
            stage1_output = self.stage1_nets[category_idx](x[i].unsqueeze(0))
            stage1_outputs[i, category_idx] = stage1_output

        
        x = F.relu(self.stage2_fc1(stage1_outputs))
        # x = self.stage2_dropout(x)
        # x = self.stage2_fc2(x)
        return x

StartingNetwork = MultiStageNet

if __name__ == "__main__":
    print("Starting Network Debug...")
    # new_network = StartingNetwork()
    # summary(new_network, (1, 24, 24))