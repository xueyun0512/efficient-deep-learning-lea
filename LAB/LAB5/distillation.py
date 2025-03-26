import torch.nn.functional as F
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sys
import torch.nn.utils.prune as prune
import sys
sys.path.append("../LAB1/")  # Ajouter le dossier parent au path
import wandb
wandb.login()
import os

checkpoint = torch.load(os.path.join("..", "LAB1", "test.pth"), map_location="cuda")
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
    
class mini_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, groups=1):
        super().__init__()  # ✅ Correction ici
        self.in_planes = 32  # Commence avec une largeur réduite

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2, groups=groups)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

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

def compute_accuracy(model, dataloader, device):
    model.eval()  # Mode évaluation
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total  # % d'accuracy


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
data_path = '/opt/img/effdl-cifar10/'
batch_size = 64
c10train = CIFAR10(data_path,train=True,download=True,transform=transform_train)
c10test = CIFAR10(data_path,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=batch_size,shuffle=True)
testloader = DataLoader(c10test,batch_size=batch_size)


# Charger les modèles ResNet18 et mini_ResNet18 avec grouped convolutions
teacher = ResNet(BasicBlock, [2, 2, 2, 2], groups=1)
student = mini_ResNet(BasicBlock, [2, 2, 2, 2], groups=8)
#Enlever keys contenant total dans state_dict
#checkpoint['model_state_dict'] = {k: v for k, v in checkpoint['model_state_dict'].items() if 'total' not in k}
    
teacher.load_state_dict(checkpoint['model_state_dict'])
teacher.eval()  # Mode évaluation

# charger le checkpoint du student si existant
if os.path.exists("mini_resnet18_distilled.pth"):
    student.load_state_dict(torch.load("mini_resnet18_distilled.pth"))
    student.eval()  # Mode évaluation

class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.T = T  # Température
        self.alpha = alpha  # Poids entre CrossEntropy et KL Divergence

    def forward(self, student_logits, teacher_logits, labels):
        # 1️⃣ CrossEntropy classique pour le student
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # 2️⃣ KL Divergence entre student et teacher (logits lissés par T)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean"
        ) * (self.T ** 2)  # On multiplie par T^2 comme dans la paper de Hinton

        # 3️⃣ Combinaison des deux pertes
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return loss


# Hyperparamètres
num_epochs = 5
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger les modèles sur GPU
teacher.to(device)
student.to(device)

# Optimizer & Loss
optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
distillation_loss = DistillationLoss(T=4.0, alpha=0.5)

# DataLoader (exemple avec CIFAR-10)
train_loader = trainloader
test_loader = testloader

# Training Loop avec affichage détaillé
for epoch in range(num_epochs):
    student.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        with torch.no_grad():  # On désactive les gradients pour le teacher
            teacher_logits = teacher(images)
        student_logits = student(images)

        # Compute Loss
        loss = distillation_loss(student_logits, teacher_logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 📌 Calcul de l'accuracy après chaque epoch
    train_acc_student = compute_accuracy(student, train_loader, device)
    test_acc_student = compute_accuracy(student, test_loader, device)
    test_acc_teacher = compute_accuracy(teacher, test_loader, device)

    # 📊 Affichage des résultats
    print(f"📢 Epoch [{epoch+1}/{num_epochs}]")
    print(f"   🟢 Loss du Student: {total_loss/len(train_loader):.4f}")
    print(f"   🔵 Accuracy Teacher (Test) : {test_acc_teacher:.2f}%")
    print(f"   🟣 Accuracy Student (Train) : {train_acc_student:.2f}%")
    print(f"   🟡 Accuracy Student (Test)  : {test_acc_student:.2f}%")
    print("-" * 50)

# Sauvegarde du modèle
torch.save(student.state_dict(), "mini_resnet18_distilled.pth")
print("✅ Modèle mini_ResNet18 distillé sauvegardé !")


