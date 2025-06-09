import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import numpy as np
import cv2

import numbers
from einops import rearrange


###############################
class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf*2)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upconv1_ill = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2_ill = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.HRconv_ill = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last_ill = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.upconv1_ill2 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2_ill2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.HRconv_ill2 = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last_ill2 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.trans1=TransformerBlock(64, 4, 2.66, False, 'WithBias').to('cuda')
        self.trans2=TransformerBlock2(64, 4, 2.66, False, 'WithBias').to('cuda')

    def forward(self, x1, x2):
        batch_size=x1.shape[0]

        L1_fea_1 = self.lrelu(self.conv_first_1(x1))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        L2_fea_1 = self.lrelu(self.conv_first_1(x2))
        L2_fea_2 = self.lrelu(self.conv_first_2(L2_fea_1))
        L2_fea_3 = self.lrelu(self.conv_first_3(L2_fea_2))

        L1_fea_3=self.trans1(L1_fea_3)+self.trans2(L1_fea_3, L2_fea_3)
        L2_fea_3=self.trans1(L2_fea_3)+self.trans2(L2_fea_3, L1_fea_3)
        channel=L1_fea_3.shape[1]
        L1_fea=torch.cat([L1_fea_3, L2_fea_3], dim=1)

        fea = self.feature_extraction(L1_fea)
        out_fea1=fea[:, 0:channel, :, :]
        out_fea2=fea[:, channel:channel*2, :, :]

        out_ill1 = torch.cat([out_fea1, L1_fea_3], dim=1)
        out_ill1 = self.lrelu(self.pixel_shuffle(self.upconv1_ill(out_ill1)))
        out_ill1 = torch.cat([out_ill1, L1_fea_2], dim=1)
        out_ill1 = self.lrelu(self.pixel_shuffle(self.upconv2_ill(out_ill1)))
        out_ill1 = torch.cat([out_ill1, L1_fea_1], dim=1)
        out_ill1 = self.lrelu(self.HRconv_ill(out_ill1))
        out_ill1 = self.conv_last_ill(out_ill1)

        out_ill12 = torch.cat([out_fea1, L1_fea_3], dim=1)
        out_ill12 = self.lrelu(self.pixel_shuffle(self.upconv1_ill2(out_ill12)))
        out_ill12 = torch.cat([out_ill12, L1_fea_2], dim=1)
        out_ill12 = self.lrelu(self.pixel_shuffle(self.upconv2_ill2(out_ill12)))
        out_ill12 = torch.cat([out_ill12, L1_fea_1], dim=1)
        out_ill12 = self.lrelu(self.HRconv_ill2(out_ill12))
        out_ill12 = self.conv_last_ill2(out_ill12)

        out_ill1_rgb_trans=nn.Sigmoid()(out_ill1)
        out_ill1_ill_trans=nn.Sigmoid()(out_ill12)
        
        out1_trans=out_ill1_rgb_trans*out_ill1_ill_trans

        out_ill2 = torch.cat([out_fea2, L2_fea_3], dim=1)
        out_ill2 = self.lrelu(self.pixel_shuffle(self.upconv1_ill(out_ill2)))
        out_ill2 = torch.cat([out_ill2, L2_fea_2], dim=1)
        out_ill2 = self.lrelu(self.pixel_shuffle(self.upconv2_ill(out_ill2)))
        out_ill2 = torch.cat([out_ill2, L2_fea_1], dim=1)
        out_ill2 = self.lrelu(self.HRconv_ill(out_ill2))
        out_ill2 = self.conv_last_ill(out_ill2)

        out_ill22 = torch.cat([out_fea2, L2_fea_3], dim=1)
        out_ill22 = self.lrelu(self.pixel_shuffle(self.upconv1_ill2(out_ill22)))
        out_ill22 = torch.cat([out_ill22, L2_fea_2], dim=1)
        out_ill22 = self.lrelu(self.pixel_shuffle(self.upconv2_ill2(out_ill22)))
        out_ill22 = torch.cat([out_ill22, L2_fea_1], dim=1)
        out_ill22 = self.lrelu(self.HRconv_ill2(out_ill22))
        out_ill22 = self.conv_last_ill2(out_ill22)
        out_ill2_rgb_trans=nn.Sigmoid()(out_ill2)
        out_ill2_ill_trans=nn.Sigmoid()(out_ill22)

        out2_trans=out_ill2_rgb_trans*out_ill2_ill_trans
        
        return out_ill1_rgb_trans, out_ill2_rgb_trans, \
            out1_trans, out2_trans, out_ill1_ill_trans, out_ill2_ill_trans


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qkv2 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qkv2_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, x2):
        b,c,h,w = x.shape

        q = self.qkv_dwconv(self.qkv(x))
        kv=self.qkv2_dwconv(self.qkv2(x2))
        k,v = kv.chunk(2, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock2, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention2(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, x2):
        x = x + self.attn(self.norm1(x), self.norm1(x2))
        x = x + self.ffn(self.norm2(x))
        return x

