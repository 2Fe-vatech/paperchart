import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import torch.optim as optim
from datasets import numberDataset
from model import Net

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = numberDataset("datasets/numbers/train", transform=transform)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=11, shuffle=True, num_workers=2
)

testset = numberDataset("datasets/numbers/valid", transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=11, shuffle=False, num_workers=2
)

classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

if os.path.isfile("./cifar_net.pth"):
    net.load_state_dict(torch.load("cifar_net.pth"))

for epoch in range(20):  # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    # net.train()
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs.float())

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

            correct = 0
            for data in testloader:
                inputs, labels = data
                outputs = net(inputs.float())
                pred = torch.argmax(outputs, dim=1)
                correct += torch.sum(torch.eq(pred, labels))
            accuracy = torch.true_divide(correct, len(testset))
            print(
                "val : {:2.2f}% ({}/{})".format(
                    accuracy.item() * 100, correct, len(testset)
                )
            )

print("Finished Training")

PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)
