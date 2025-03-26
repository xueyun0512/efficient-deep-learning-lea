import torch
import torchvision.models as models
from torchinfo import summary
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import sys
import math
from collections import OrderedDict
import torch.nn.utils.prune as prune
sys.path.append("../LAB1")
from resnet import ResNet18

from src.factorization_train import ResNet18_Modified_Depthwise

# Fetch the model
loaded_cpt = torch.load('./models/simple_depthwise_local_20_structured.pth')
# print(loaded_cpt['model_state_dict'])

is_pruned = True
# if the model has been pruned then change the dictionary keys name to remove _orig and _mask
# Iterate through the loaded state dictionary
if is_pruned:
    new_state_dict = OrderedDict()
    for key, value in loaded_cpt['model_state_dict'].items():
        if key.endswith('_orig'):
            # Get the base key (without '_orig' suffix)
            base_key = key[:-5]

            # Get the corresponding mask key
            mask_key = base_key + '_mask'

            # Retrieve the original weights and mask
            orig_weights = value
            mask = loaded_cpt['model_state_dict'][mask_key]

            # Apply the mask to the original weights
            final_weights = orig_weights * mask

            # Store the final weights in the new state dictionary
            new_state_dict[base_key] = final_weights
        elif key.endswith('_mask'):
            # Skip mask keys, as they are already processed
            continue
        else:
            # For keys without '_orig' or '_mask', add them directly
            new_state_dict[key] = value

model_no_flavor = ResNet18()
model = ResNet18_Modified_Depthwise(model_no_flavor, use_depthwise_layers=['layer3', 'layer4'])
#model = ResNet18_Modified_Depthwise(model_no_flavor, True)
#model = ResNet18_Modified_Grouped(model_no_flavor, True, 4)
#model.load_state_dict(loaded_cpt['model_state_dict'])
#model.load_state_dict(new_state_dict, strict=False)


# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

# if the modelhas been pruned, return the number of 0
zero_params = sum((p==0.).sum().item() for p in model.parameters())
print(f"Total number of 0: {zero_params}")

#print(f"Number of trainable parameters: {trainable_params}")
# Calculate the number of FLOPs
input_tensor = torch.randn(1, 3, 32, 32) # Example input for ImageNet models
flops = FlopCountAnalysis(model, input_tensor)
print(f"Total number of FLOPs: {flops.total()}")

