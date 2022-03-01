import torch
import torch.nn as nn 
from fast_transformers.builders import TransformerEncoderBuilder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class VTP(nn.Module):
    

    @staticmethod
    def modify_commandline_options(parser):

        parser.add_argument('--vtp_patch_size', type=int, default=1, nargs = "+", help='vit patch size for cnn feature') 
        parser.add_argument('--vtp_scale', type=int, default=8, help='scale for cnn feature') 
        parser.add_argument('--vtp_dim', type=int, default=512, help='feature dim for vtp') 

    def __init__(self, opt):
        super(VTP, self).__init__()
        self.opt = opt
        patch_size = self.opt.vtp_patch_size
        image_size = int(self.opt.upsample_size / self.opt.vtp_scale)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 64 * self.opt.vtp_scale * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, self.opt.vtp_dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , self.opt.vtp_dim))

        # self.dropout = nn.Dropout(0.25)
        # builder = TransformerEncoderBuilder.from_kwargs(
        # n_layers=4,
        # n_heads=8,
        # query_dimensions= int(self.opt.vtp_dim / 8),
        # value_dimensions=int(self.opt.vtp_dim / 8),
        # feed_forward_dimensions=self.opt.vtp_dim
        # )

        # builder.attention_type = "linear"
        # self.transformer = builder.get()

        # self.Q = nn.Parameter(torch.ones(self.opt.vtp_dim), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(self.opt.vtp_dim), requires_grad=True)

        self.attend = nn.Softmax(dim = 1)
        self.norm1 = nn.LayerNorm(self.opt.vtp_dim, elementwise_affine = True)
        self.norm2 = nn.LayerNorm(self.opt.vtp_dim, elementwise_affine = False)

        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x):
        x = self.to_patch_embedding(x)
        # x += self.pos_embedding
        # x = self.dropout(x)
        # x = self.transformer(x)
        x = self.norm1(x)
        # dots = torch.mul(x, self.Q) + self.bias
        dots = torch.sum(x, dim = -1, keepdim = True) # sum after norm , norm weight * dim feature
        # print(dots[0][:20] , torch.max(dots[0]))
        # print(self.attend(dots[0][:20]), torch.max(self.attend(dots[0])), torch.argmax(self.attend(dots[0])))

        # print(torch.max(self.Q), torch.argmax(self.Q), torch.min(self.Q), torch.argmin(self.Q), torch.topk(self.Q, 10, dim = 0))
        attn = torch.transpose(self.attend(dots),1,2)
        # print(attn[0])
        out = torch.matmul(attn, x)
        out = torch.squeeze(out, dim = 1)
        # out = self.norm2(out)
        attn = attn.view(attn.shape[0],1,self.image_height, self.image_width)
        # print(attn)
        return {"feature":out,"attn":attn}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.vtp_patch_size = 1
    parser.vtp_scale = 32
    parser.vtp_dim = 2048
    parser.upsample_size = 448

    input = torch.randn(32,2048,14,14)
    vtp = VTP(parser)
    output = vtp(input)
    print(output["feature"].shape)