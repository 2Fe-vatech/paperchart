import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import numberDataset
from model import Net

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

model = Net()
model.load_state_dict(torch.load("cifar_net.pth"))
model.eval()

testset = numberDataset("datasets/numbers/valid", train=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=11, shuffle=False, num_workers=2
)

count = 0
for data in testloader:
    inputs, labels = data
    outputs = model(inputs.float())
    pred = torch.argmax(outputs, dim=1)
    count += torch.sum(pred == labels)

print(count)
