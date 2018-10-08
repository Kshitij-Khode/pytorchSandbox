import torch
import torch.nn
import torch.nn.functional as tfunc

class LeNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, opTensor):
        opTensor = tfunc.max_pool2d(tfunc.relu(self.conv1(opTensor)), (2, 2))
        opTensor = tfunc.max_pool2d(tfunc.relu(self.conv2(opTensor)), (2, 2))
        opTensor = opTensor.view(-1, self.num_flat_features(opTensor))
        opTensor = tfunc.relu(self.fc1(opTensor))
        opTensor = tfunc.relu(self.fc2(opTensor))
        opTensor = self.fc3(opTensor)

        return opTensor

    def num_flat_features(self, tensor):
        size = tensor.size()[1:]
        numFeatures = 1
        for s in size:
            numFeatures *= s

        return numFeatures

leNet  = LeNet()
input  = torch.randn(1, 1, 32, 32)
output = leNet(input)
target = torch.randn(10).view(1, -1)
loss   = torch.nn.MSELoss()(output, target)
optimizer = torch.optim.SGD(leNet.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()