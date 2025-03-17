import torch.nn.functional as F
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import wandb
from thop import profile
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sys

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, groups=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, groups=groups)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, groups):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups=groups))
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

# Fonction pour instancier ResNet18 avec le paramètre "groups"
def ResNet18(groups=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], groups=groups)



def get_dataloaders(batch_size):
    rootdir = '/opt/img/effdl-cifar10/'
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    trainset = torchvision.datasets.CIFAR10(root=rootdir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=rootdir, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def train(args):
    # Initialize Weights & Biases
    wandb.init(project="deep-learning-lab3", config=args.__dict__ , name="group_8_lea", job_type="training_test")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preprocessing
    normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize_scratch,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch,
    ])

    c10train = CIFAR10(args.data_path,train=True,download=True,transform=transform_train)
    c10test = CIFAR10(args.data_path,train=False,download=True,transform=transform_test)

    trainloader = DataLoader(c10train,batch_size=args.batch_size,shuffle=True)
    testloader = DataLoader(c10test,batch_size=args.batch_size)

    # Initialize model, loss, optimizer
    model = ResNet18(groups=8)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Training accuracy
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(trainloader)

        print(f"TRAINING: Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Learning rate: {optimizer.param_groups[0]['lr']}")

        # Evaluate on test set
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                correct_test += predicted.eq(labels).sum().item()
                total_test += labels.size(0)

        test_accuracy = 100 * (correct_test / total_test)
        test_loss= test_loss / len(testloader)

        scheduler.step(test_loss / len(testloader))

        # number of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # number operations
        input_tensor = torch.randn(1, 3, 32, 32).to(device)
        flops, _ = profile(model, inputs=(input_tensor,))
        
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_learning_rate": optimizer.param_groups[0]['lr'],
            "num_params": num_params,
            "num_operations": flops
        })
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_accuracy:.2f}% "
              f"Learning Rate: {optimizer.param_groups[0]['lr']}"
              f"Num params: {num_params}"
              f"Num operations: {flops}"
              
              )
        
        
        
    print("Training complete my boss")
    wandb.finish()
    # Save model along with training hyperparameters
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
    }

    torch.save(checkpoint, args.save_path)
    print(f"Model and training details saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for regularization")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_path", type=str, default="/opt/img/effdl-cifar10/", help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="./group_Resnet.pth", help="Path to save the model")
    


    args = parser.parse_args()
    
    train(args)

