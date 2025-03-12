import torch
import torchvision.models as models
from torchinfo import summary
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import sys
import math
import torch.nn.utils.prune as prune
sys.path.append("../LAB1")
from resnet import ResNet18

# Depthwise Separable Convolution
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
    def __init__(self, original_model, use_depthwise):
        super(ResNet18_Modified_Depthwise, self).__init__()
        self.use_depthwise = use_depthwise
        self.modified_model = self.modify_layers(original_model)

    def modify_layers(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1 and self.use_depthwise:
                depthwise_conv = DepthwiseSeparableConv(module.in_channels, module.out_channels,
                                                        module.kernel_size, module.stride,
                                                        module.padding, module.bias is not None)
                parent_name = name.rsplit('.', 1)[0]
                
                if '.' in name:  # If it's inside a block
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, name.split('.')[-1], depthwise_conv)
                else:  # If it's a top-level module
                    setattr(model, name, depthwise_conv)
        return model

    def forward(self, x):
        return self.modified_model(x)


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

# Fetch the model
loaded_cpt = torch.load('./models/grouped_4.pth')
# print(loaded_cpt['model_state_dict'])
model_no_flavor = ResNet18()
#model = ResNet18_Modified_Depthwise(model_no_flavor, True)
model = ResNet18_Modified_Grouped(model_no_flavor, True, 4)
model.load_state_dict(loaded_cpt['model_state_dict'])


# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
#print(f"Number of trainable parameters: {trainable_params}")
# Calculate the number of FLOPs
input_tensor = torch.randn(1, 3, 32, 32) # Example input for ImageNet models
flops = FlopCountAnalysis(model, input_tensor)
print(f"Total number of FLOPs: {flops.total()}")

