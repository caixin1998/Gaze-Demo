import torch
import torch.nn as nn 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ATP(nn.Module):

    def __init__(self, patch_size, patch_dim, vtp_dim):
        super(ATP, self).__init__()
        # print("ATP", patch_size)
        patch_height, patch_width = pair(patch_size)
      
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1),
            nn.Linear(patch_dim, vtp_dim),
        )

        self.attend = nn.Softmax(dim = 1)
        self.norm1 = nn.LayerNorm(vtp_dim, elementwise_affine = True)
        self.norm2 = nn.LayerNorm(vtp_dim, elementwise_affine = False)

        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, x):
        x = self.to_patch_embedding(x)
 
        x = self.norm1(x)
        dots = torch.sum(x, dim = -1, keepdim = True)

        attn = torch.transpose(self.attend(dots),1,2)
        out = torch.matmul(attn, x)
        out = torch.squeeze(out, dim = 1)
        # out = self.norm2(out)
        attn = attn.view(attn.shape[0],1,self.patch_height, self.patch_width)
        return {"feature":out,"attn":attn}
