import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_grouped=False, groups=2):
        super(BasicBlock, self).__init__()
        # Use grouped convolution if use_grouped is True
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, 
            bias=False, groups=groups if use_grouped else 1
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, 
            bias=False, groups=groups if use_grouped else 1
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, 
                    stride=stride, bias=False, groups=groups if use_grouped else 1
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_grouped=False, groups=2):
        super(Bottleneck, self).__init__()
        # Use grouped convolution if use_grouped is True
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, 
            bias=False, groups=groups if use_grouped else 1
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, 
            bias=False, groups=groups if use_grouped else 1
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, 
            bias=False, groups=groups if use_grouped else 1
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, 
                    stride=stride, bias=False, groups=groups if use_grouped else 1
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_grouped=False, groups=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # First layer: Always use standard convolution (skip factorization)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Apply grouped convolutions to the rest of the network
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_grouped=use_grouped, groups=groups)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_grouped=use_grouped, groups=groups)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_grouped=use_grouped, groups=groups)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_grouped=use_grouped, groups=groups)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, use_grouped=False, groups=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_grouped, groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(use_grouped=False, groups=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_grouped=use_grouped, groups=groups)


def ResNet34(use_grouped=False, groups=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], use_grouped=use_grouped, groups=groups)


def ResNet50(use_grouped=False, groups=2):
    return ResNet(Bottleneck, [3, 4, 6, 3], use_grouped=use_grouped, groups=groups)


def ResNet101(use_grouped=False, groups=2):
    return ResNet(Bottleneck, [3, 4, 23, 3], use_grouped=use_grouped, groups=groups)


def ResNet152(use_grouped=False, groups=2):
    return ResNet(Bottleneck, [3, 8, 36, 3], use_grouped=use_grouped, groups=groups)


def test():
    # Example usage with grouped convolutions (skipping the first layer)
    net = ResNet18(use_grouped=True, groups=4)  # groups=4 is now allowed
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())  # Expected output: torch.Size([1, 10])

# test()