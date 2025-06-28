import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out




class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=6):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(2)
        self.fc = nn.Linear(512 * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.reshape(-1,1,14100)

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)


        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class irCNN(nn.Module):
    expansion = 1

    def __init__(self,num_classes, stride=3,):
        super(irCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 31, kernel_size=11, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(31)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(31, 62, kernel_size=11, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(62)
        self.Maxpool = nn.MaxPool1d(2,2)


        self.dropout = nn.Dropout(0.48599)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(62*390, 4927)
        self.fc2 = nn.Linear(4927, 2785)
        self.fc3 = nn.Linear(2785, 1574)
        self.fc4 = nn.Linear(1574,num_classes)




    def forward(self, x):
        x = x.reshape(-1, 1, 14100)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.Maxpool(out)
        # 31,7046
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.Maxpool(out)
        #62,1654
        # print(out.shape)
        out = torch.flatten(out,1)
        # print(out.shape)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.sigmoid(out)


        return out

def resnet50(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2,2 , 2, 2], num_classes)
if __name__ == "__main__":
    data = torch.rand((32,1,13272))
    model=resnet50(6)
    y = model.forward(data)
    # m = nn.AdaptiveAvgPool1d(5)
    # input = torch.randn(1, 64, 8)
    # output = m(input)
    # print(output.shape)
