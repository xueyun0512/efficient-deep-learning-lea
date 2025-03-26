import argparse
import math
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sys
import torch.nn.utils.prune as prune

sys.path.append("../LAB1")
from resnet_bis import ResNet18

############# FACTORIZATION: DEPTH WISE ########################################################################################################################
# Depthwise Separable Convolution
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
#                                    stride=stride, padding=padding, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
    
# class ResNet18_Modified_Depthwise(nn.Module):
#     def __init__(self, original_model, use_depthwise):
#         super(ResNet18_Modified_Depthwise, self).__init__()
#         self.use_depthwise = use_depthwise
#         self.modified_model = self.modify_layers(original_model)

#     def modify_layers(self, model):
#         for name, module in model.named_modules():
#             if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1 and self.use_depthwise:
#                 depthwise_conv = DepthwiseSeparableConv(module.in_channels, module.out_channels,
#                                                         module.kernel_size, module.stride,
#                                                         module.padding, module.bias is not None)
#                 parent_name = name.rsplit('.', 1)[0]
                
#                 if '.' in name:  # If it's inside a block
#                     parent = dict(model.named_modules())[parent_name]
#                     setattr(parent, name.split('.')[-1], depthwise_conv)
#                 else:  # If it's a top-level module
#                     setattr(model, name, depthwise_conv)
#         return model

#     def forward(self, x):
#         return self.modified_model(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResNet18_Modified_Depthwise(nn.Module):
    def __init__(self, original_model, use_depthwise_layers=['layer3', 'layer4']):
        super(ResNet18_Modified_Depthwise, self).__init__()
        self.use_depthwise_layers = use_depthwise_layers
        self.modified_model = self.modify_layers(original_model)

    def modify_layers(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
                for layer in self.use_depthwise_layers:
                    if layer in name:
                        depthwise_conv = DepthwiseSeparableConv(module.in_channels, module.out_channels,
                                                                module.kernel_size, module.stride,
                                                                module.padding, module.bias is not None)
                        parent_name = name.rsplit('.', 1)[0]
                        if '.' in name:
                            parent = dict(model.named_modules())[parent_name]
                            setattr(parent, name.split('.')[-1], depthwise_conv)
                        else:
                            setattr(model, name, depthwise_conv)
        return model

    def forward(self, x):
        return self.modified_model(x)

############# FACTORIZATION: GROUP 2, 4, 8, etc ########################################################################################################################
class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=2):
        super(GroupedConv2d, self).__init__()
        # Ensure that in_channels and out_channels are divisible by groups
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        
        self.groups = groups
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ResNet18_Modified_Grouped(nn.Module):
    def __init__(self, original_model, use_grouped, groups=2):
        super(ResNet18_Modified_Grouped, self).__init__()
        self.use_grouped = use_grouped
        self.groups = groups
        self.modified_model = self.modify_layers(original_model)

    def modify_layers(self, model):
        for name, module in model.named_modules():
            # Replace standard Conv2d with GroupedConv2d if kernel_size > 1 and use_grouped is True
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1 and self.use_grouped:
                # Check if groups is compatible with in_channels and out_channels
                if module.in_channels % self.groups == 0 and module.out_channels % self.groups == 0:
                    # Create the grouped convolution layer
                    grouped_conv = GroupedConv2d(module.in_channels, module.out_channels,
                                                module.kernel_size, module.stride,
                                                module.padding, module.bias is not None,
                                                groups=self.groups)
                    
                    # Replace the original Conv2d layer with the grouped convolution
                    parent_name = name.rsplit('.', 1)[0]
                    if '.' in name:  # If it's inside a block
                        parent = dict(model.named_modules())[parent_name]
                        setattr(parent, name.split('.')[-1], grouped_conv)
                    else:  # If it's a top-level module
                        setattr(model, name, grouped_conv)
                else:
                    # Calculate the greatest common divisor (GCD) for a compatible groups
                    gcd = math.gcd(module.in_channels, module.out_channels)
                    if gcd > 1:
                        # Use the GCD as the groups
                        adjusted_groups = gcd
                        grouped_conv = GroupedConv2d(module.in_channels, module.out_channels,
                                                    module.kernel_size, module.stride,
                                                    module.padding, module.bias is not None,
                                                    groups=adjusted_groups)
                        
                        # Replace the original Conv2d layer with the adjusted grouped convolution
                        parent_name = name.rsplit('.', 1)[0]
                        if '.' in name:  # If it's inside a block
                            parent = dict(model.named_modules())[parent_name]
                            setattr(parent, name.split('.')[-1], grouped_conv)
                        else:  # If it's a top-level module
                            setattr(model, name, grouped_conv)
                        
                        print(
                            f"Adjusted groups to {adjusted_groups} for layer {name} "
                            f"(in_channels={module.in_channels}, out_channels={module.out_channels})"
                        )
                    else:
                        # Skip the layer if no compatible groups is found
                        print(
                            f"Skipping layer {name} because no compatible groups found "
                            f"(in_channels={module.in_channels}, out_channels={module.out_channels})"
                        )
        return model

    def forward(self, x):
        return self.modified_model(x)

#############################################################################################################################################################
def factorization_train(args):
    # Data preprocessing
    normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize_scratch,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch,
    ])
    c10train = CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
    c10test = CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(c10train, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(c10test, batch_size=args.batch_size)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fetch the model and modify it
    # loaded_cpt = torch.load('./models/test.pth')
    # original_model = ResNet18()
    # original_model.load_state_dict(loaded_cpt['model_state_dict'])
    # model = ResNet18_Modified_Depthwise(original_model, args.use_depthwise).to(device)

    # use the original model from kuangliu repo
    model_no_falvor = ResNet18()

    # Depthwise total
    # model = ResNet18_Modified_Depthwise(model_no_falvor, args.use_depthwise).to(device)

    # Depthwise partial
    model = ResNet18_Modified_Depthwise(model_no_falvor, use_depthwise_layers=['layer3', 'layer4']).to(device)

    # Grouped factorization
    # model = ResNet18(use_grouped=args.use_grouped, groups=args.groups).to(device)
    # model = ResNet18_Modified_Grouped(model_no_falvor, args.use_grouped, args.groups).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Initialize Weights & Biases
    #name = f"factorisation_grouped_{args.groups}"
    name = "depthwise_partial"
    wandb.init(project="deep-learning-lab3", config=args.__dict__, name=name, job_type="training_test")

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
        test_loss = test_loss / len(testloader)

        scheduler.step(test_loss)
        
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_learning_rate": optimizer.param_groups[0]['lr'],
        })
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_accuracy:.2f}% "
              f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    
    print("Training complete my boss")
    wandb.finish()
    
    model.half()
    # Save model
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for regularization")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_path", type=str, default="/opt/img/effdl-cifar10/", help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="./models/depthwise_partial.pth", help="Path to save the model")
    parser.add_argument("--use_depthwise", action='store_true', help="Use depthwise separable convolutions")
    parser.add_argument("--use_grouped", action='store_true', help="Use grouped factorization on convolutions")
    parser.add_argument("--groups", type=int, default=2, help="Group size for grouped factorization")

    args = parser.parse_args()
    factorization_train(args)
