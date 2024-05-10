import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
#import wget
os.environ['TORCH_HOME'] = '../../pretrained_models'
from timm.models.layers import to_2tuple,trunc_normal_

class CustomTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class CustomTransformerModel(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([CustomTransformerBlock(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_blocks)])
        # Assuming the presence of a class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
       # self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
        num_patches = 1212#(img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        #self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=1000)
        new_pos_embed = nn.Parameter(torch.zeros(1, 3084 + 1, 768))
        self.pos_embed = new_pos_embed
        trunc_normal_(self.pos_embed, std=.02)
      #  self.mlp_head = nn.Sequential(nn.LayerNorm(d_model))
        self.lin_input = nn.Linear(256, 768)
       # self.conv = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))

    def forward(self, x):
        # Implement forward pass considering the cls token and positional embeddings
        batch_size, num_patches = x.shape[:2]

        #print(x)
        x = self.lin_input(x)
        
        # Reshape x to interpret the flattened patches in 2D
       # x = x.view(batch_size, num_patches, 16, 16)
        
        # Merge batch and num_patches dimensions and add a channel dimension for the convolution
       # x = x.view(-1, 1, 16, 16)
        
        # Apply convolution, assume self.conv is defined elsewhere
       # x = self.conv(x)
        
        # Flatten the convolution output from dimension 2 onwards
        # This step depends on the output shape of your convolution. 
        # Flatten(2) will flatten all dimensions starting from the third dimension into a single dimension.
       # x = x.flatten(2)
        
        # Transpose the second and third dimensions
        # Assuming after flattening, x has shape [batch_size*num_patches, channels, flattened_conv_output]
        # Transpose to move the channels dimension to the end
       # x = x.transpose(1, 2)
        
        # Now, we knoxw the flattened size of each conv output. Let's reshape x back to include batch_size and num_patches
       # x = x.view(batch_size, num_patches, 768)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Use self.cls_token
      #  dist_token = self.dist_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
       # x = self.pos_enc(x)

        song_size = x.shape[1]
        x = x + self.pos_embed[:,:song_size]

        for block in self.blocks:
           # print(x)
            x = block(x)
            
        x = x[:, 0]
       # x = self.mlp_head(x)
        return x


def map_pretrained_keys_to_model(pretrained_keys):
    # Initialize an empty dictionary to hold the mapping
    mapped_keys = {}

    # Map the global components like cls_token, pos_embed, etc.
    global_mappings = {
        'cls_token': 'cls_token',  # Adjust if your custom model uses a different key
       # 'pos_embed': 'pos_embed',  # Adjust based on your custom model's key for positional embeddings
        'dist_token': 'dist_token',  # Include this if your model uses a distillation token
        # Add mappings for patch embeddings if they're named differently in your model
        'patch_embed.proj.weight': 'conv.weight',
        'patch_embed.proj.bias': 'conv.bias',
    }

    # Assuming all blocks need similar mapping adjustments
    for block_num in range(12):  # Adjust the range if your model has a different number of blocks
        block_mappings = {
            f'blocks.{block_num}.norm1.weight': f'blocks.{block_num}.norm1.weight',
            f'blocks.{block_num}.norm1.bias': f'blocks.{block_num}.norm1.bias',
            f'blocks.{block_num}.norm2.weight': f'blocks.{block_num}.norm2.weight',
            f'blocks.{block_num}.norm2.bias': f'blocks.{block_num}.norm2.bias',
            f'blocks.{block_num}.attn.qkv.weight': f'blocks.{block_num}.attn.in_proj_weight',
            f'blocks.{block_num}.attn.qkv.bias': f'blocks.{block_num}.attn.in_proj_bias',
            f'blocks.{block_num}.attn.proj.weight': f'blocks.{block_num}.attn.out_proj.weight',
            f'blocks.{block_num}.attn.proj.bias': f'blocks.{block_num}.attn.out_proj.bias',
            f'blocks.{block_num}.mlp.fc1.weight': f'blocks.{block_num}.mlp.0.weight',
            f'blocks.{block_num}.mlp.fc1.bias': f'blocks.{block_num}.mlp.0.bias',
            f'blocks.{block_num}.mlp.fc2.weight': f'blocks.{block_num}.mlp.3.weight',
            f'blocks.{block_num}.mlp.fc2.bias': f'blocks.{block_num}.mlp.3.bias',
        }
        mapped_keys.update(block_mappings)

    # Include mappings for the normalization layer and head(s) at the end of the model, if applicable
    model_end_mappings = {
        'mlp_head.0.weight': 'mlp_head.0.weight',
        'mlp_head.0.bias': 'mlp_head.0.bias',
        'mlp_head.1.weight': 'mlp_head.1.weight',
        'mlp_head.1.bias': 'mlp_head.1.bias',
    }

    # Update the global mapping dictionary with block and model end mappings
    mapped_keys.update(global_mappings)
    mapped_keys.update(model_end_mappings)

    # Adjust the mapping based on the provided pretrained_keys
    adjusted_keys = {pretrained_key: mapped_keys[pretrained_key] for pretrained_key in pretrained_keys if pretrained_key in mapped_keys}

    return adjusted_keys


def adjust_pretrained_dict_keys(pretrained_dict, adjusted_keys):
    new_dict = {}
    for k, v in pretrained_dict.items():
        new_key = adjusted_keys.get(k, None)
        if new_key:
            new_dict[new_key] = v
        else:
            print(f"Key not found in mapping or not needed for adjustment: {k}")
    return new_dict


def load_custom_weights(model, pre_trained_weights):

    pretrained_dict = torch.load(pre_trained_weights, map_location='cuda')
    print(pretrained_dict.keys())
    # Prepare a new state dictionary with adjusted keys
    # Load the pre-trained state dictionary

    # Adjust keys in the pre-trained state dict
    adjusted_dict = {k.replace('module.v.', ''): v for k, v in pretrained_dict.items()}  # Use this if your model does not use 'module.' prefix
    #print("keyeeye", adjusted_dict.keys())
    adjusted_keys = map_pretrained_keys_to_model(adjusted_dict.keys())
    adjusted_pretrained_dict = adjust_pretrained_dict_keys(adjusted_dict, adjusted_keys)
   # print(adjusted_keys.keys())
    
    # If your model actually expects 'module.' prefix but the keys don't have it (less common scenario)
    # adjusted_dict = {'module.' + k if not k.startswith('module.') else k: v for k, v in pretrained_dict.items()}

   # adjusted_pretrained_dict['mlp_head.0.weight'] = pretrained_dict['module.mlp_head.0.weight']
   # adjusted_pretrained_dict['mlp_head.0.bias'] = pretrained_dict['module.mlp_head.0.bias']
 #   adjusted_pretrained_dict['mlp_head.1.weight'] = pretrained_dict['module.mlp_head.1.weight']
  #  adjusted_pretrained_dict['mlp_head.1.bias'] = pretrained_dict['module.mlp_head.1.bias']
    list1 = list(adjusted_pretrained_dict.keys())
    list2 = list(model.state_dict().keys())

    list1_filtered = [item for item in list1 if item not in list2]
    list2_filtered = [item for item in list2 if item not in list1]

    


    print(list1_filtered)
    print(list2_filtered)
    
    # Print the filtered lists

    # Load the adjusted dict into your model (ensure your model architecture matches the keys)
    model.load_state_dict(adjusted_pretrained_dict, strict=False)

    
    

    


# Example usage
#model = CustomTransformerModel(num_blocks=12, d_model=1536, n_heads=12, dim_feedforward=3072)
#load_custom_weights(model, 'pretrained_ast/audioset_10_10_0.4593.pth')