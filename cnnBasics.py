#!/home/kshitij/installedPrograms/anaconda3/bin/python

import torch, time

import torchvision            as tv
import torchvision.transforms as tt
import matplotlib.pyplot      as plt
import numpy                  as np
import torch.nn               as tnn
import torch.nn.functional    as tnnfunc
import torch.optim            as toptim

transform = tt.Compose([tt.ToTensor(),
                        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainSet = tv.datasets.CIFAR10(root='./data',
                               train=True,
                               download=True,
                               transform=transform)

trainLoader = torch.utils.data.DataLoader(trainSet,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)


testSet = tv.datasets.CIFAR10(root='./root',
                                       train=False,
                                       download=True,
                                       transform=transform)

testLoader = torch.utils.data.DataLoader(testSet,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane',
           'car',
           'bird',
           'cat',
           'deer',
           'dog',
           'frog',
           'horse',
           'ship',
           'truck')

class CustomNet(tnn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = tnn.Conv2d(3, 6, 5)
        self.pool1 = tnn.MaxPool2d(2, 2)
        self.conv2 = tnn.Conv2d(6, 16, 5)
        self.pool2 = tnn.MaxPool2d(2, 2)
        self.fc1   = tnn.Linear(16*5*5, 120)
        self.fc2   = tnn.Linear(120, 84)
        self.fc3   = tnn.Linear(84, 10)

    def forward(self, opTensor):
        opTensor = self.pool1(tnnfunc.relu(self.conv1(opTensor)))
        opTensor = self.pool2(tnnfunc.relu(self.conv2(opTensor)))
        opTensor = opTensor.view(-1, 16*5*5)
        opTensor = tnnfunc.relu(self.fc1(opTensor))
        opTensor = tnnfunc.relu(self.fc2(opTensor))
        opTensor = self.fc3(opTensor)

        return opTensor

device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
customNet = CustomNet()
lossCrit  = tnn.CrossEntropyLoss()
optimizer = toptim.SGD(customNet.parameters(), lr=0.001, momentum=0.9)
customNet.to(device)

for epoch in range(2):
    runningLoss = 0.0

    for i, data in enumerate(trainLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = customNet(inputs)
        loss    = lossCrit(outputs, labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f. resetting loss.' % (epoch+1, i+1, runningLoss/2000))
            runningLoss = 0.0

print('finished training')

testIter       = iter(testLoader)
images, labels = testIter.next()

def imshow(img):
    img   = img/2 + 0.5
    npImg = img.numpy()
    plt.imshow(np.transpose(npImg, (1, 2, 0)))

imshow(tv.utils.make_grid(images))
print('groundtruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

images       = images.to(device)
outputs      = customNet(images)
_, predicted = torch.max(outputs, 1)
print('predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct, total = 0, 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs        = customNet(images)
        _, predicted   = torch.max(outputs, 1)
        total         += labels.size()[0]
        correct       += (predicted == labels).sum().item()

print('accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

classCorrect = [0.0 for i in range(10)]
classTotal   = [0.0 for i in range(10)]

with torch.no_grad():
    for data in testLoader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs        = customNet(images)
        _, predicted   = torch.max(outputs, 1)
        c              = (predicted == labels)

        for i in range(4):
            label                = labels[i]
            classCorrect[label] += c[i].item()
            classTotal[label]   += 1

for i in range(10):
    print('accuracy of %5s : %2d %%' % (classes[i], 100*classCorrect[i]/classTotal[i]))

