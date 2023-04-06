#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:35:55 2023

@author: kui
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .unet2d import Encoder, Up, ResidualBlock
import einops

def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x


class SelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)

    def self_attn(self, x):
        BS, V, f = x.shape

        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)
        out = self.dropout2(self.fc(out))
        return out

    def forward(self, x):
        BS, V, f = x.shape
        assert f == self.f_dim

        x = x + self.self_attn(self.layer_norm(x))
        x = self.ff(x)

        return x


class InterSelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)

    def self_attn(self, x0, x1):
        BS, V, f = x0.shape

        q = self.w_qs(x0).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x1).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x1).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)
        out = self.dropout2(self.fc(out))
        return out

    def forward(self, x, y):
        BS, V, f = x.shape
        assert f == self.f_dim

        x = x + self.self_attn(self.layer_norm(x), self.layer_norm(y))
        x = self.ff(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        #print(self.num_patches)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class img_attn(nn.Module):
    def __init__(self, verts_f_dim, img_f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.verts_f_dim = verts_f_dim

        self.fc = nn.Linear(img_f_dim, verts_f_dim)
        self.Attn = SelfAttn(verts_f_dim, n_heads=n_heads, hid_dim=verts_f_dim, dropout=dropout)

    def forward(self, verts_f, img_f):

        
        V = verts_f.shape[1]
        img_f = self.fc(img_f)
        #print(verts_f.shape, img_f.shape)
        x = torch.cat([verts_f, img_f], dim=1)
        x = self.Attn(x)

        verts_f = x[:, :V]

        return verts_f


class img_ex(nn.Module):
    def __init__(self, img_size, img_f_dim,
                 grid_size, grid_f_dim,
                 verts_f_dim,
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        self.verts_f_dim = verts_f_dim
        self.encoder = img_feat_to_grid(img_size, img_f_dim, grid_size, grid_f_dim, n_heads, dropout)
        self.attn = img_attn(verts_f_dim, grid_f_dim, n_heads=n_heads, dropout=dropout)

        for m in self.modules():
            weights_init(m)

    def forward(self, img, verts_f):
        
        grid_feat = self.encoder(img)
        #print(verts_f.shape, grid_feat.shape)
        verts_f = self.attn(verts_f, grid_feat)
        return verts_f


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    
    
    
class img_feat_to_grid(nn.Module):
    def __init__(self, img_size, img_f_dim, grid_size, grid_f_dim, n_heads=4, dropout=0.01):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.img_size = img_size
        self.grid_f_dim = grid_f_dim
        self.grid_size = grid_size
        #self.proj_offest = PatchEmbed(self.img_size, self.grid_size, self.img_f_dim, 2)#nn.Conv2d(img_f_dim, grid_f_dim, kernel_size=patch_size, stride=patch_size)
        self.proj0 = PatchEmbed(self.img_size, self.grid_size, self.img_f_dim, self.grid_f_dim)#nn.Conv2d(img_f_dim, grid_f_dim, kernel_size=patch_size, stride=patch_size)
        self.proj1 = PatchEmbed(self.img_size, self.grid_size, self.img_f_dim, self.grid_f_dim)#nn.Conv2d(img_f_dim, grid_f_dim, kernel_size=patch_size, stride=patch_size)
         
        self.num_paths = self.proj1.num_patches 
        #self.position_embeddings = nn.Embedding(self.num_path, grid_f_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_paths, self.grid_f_dim))
        self.self_attn = InterSelfAttn(grid_f_dim, n_heads=n_heads, hid_dim=grid_f_dim, dropout=dropout)
        
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.img_f_dim , self.img_f_dim, 3, 1, 1), #, groups=self.n_group_channels
            LayerNormProxy(self.img_f_dim),
            nn.GELU(),
            nn.Conv2d(self.img_f_dim, 2, 1, 1, 0, bias=False)
        )        
        
        
    def forward(self, img):
        bs = img.shape[0]
        #print(img.shape, self.img_f_dim )
        offset = F.relu(self.conv_offset(img))
        #print(offset.shape)
        x_sampled = F.grid_sample(
            input=img, 
            grid=offset.permute(0,2,3,1).contiguous(), # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        #print('x_sampled', x_sampled.shape)          
        #class_pos_embed = self.position_embeddings.expand(bs, -1, -1)
        #print('!!!!', img.shape, x_sampled.shape, class_pos_embed.shape)
        #grid_feat = grid_feat + class_pos_embed#grid_offest
        x0 = self.proj0(img)
        grid_feat = self.proj1(x_sampled)
        #grid_offest = F.relu(self.proj_offest(img))
        #print('!!!!', grid_feat.shape, x0.shape)
        # print(grid_feat.shape, grid_offest.shape)
        # x_sampled = F.grid_sample(
        #     input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
        #     grid=pos[..., (1, 0)], # y, x -> x, y
        #     mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        #print('x_sampled', class_pos_embed.shape, grid_feat.shape)           
        #grid_feat = grid_feat + class_pos_embed#grid_offest
        grid_feat = self.self_attn(x0, grid_feat)
        return grid_feat
    
    



    

class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))
    
    def forward(self, x):

        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0) 
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))
        
        return x
    
    

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x
    
def make_upsample_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            Interpolate(2, 'bilinear'))
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=3,
                stride=1,
                padding=1
                ))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class PointDAT(nn.Module):

    def __init__(self, in_channels=1, out_channels=[6], img_size=(224,224), patch_size=1, expansion=4,
                 dim_stem=256, dims=[256, 128, 64, 64], depths=[1, 1, 1, 1], 
                 heads=[8, 4, 2, 1], 
                 window_sizes=[2, 2, 2, 2],
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 **kwargs):
        super().__init__()
        self.out_dim = [1024, 512, 256, 128, 128]
        self.backbone_net = Encoder(1, 256)
        #self.image_size = [(48,64), (96,128), (192,256), (384,512)]  
        self.image_size = [(64,64), (128,128), (256,256), (512,512)]          
        self.window_sizes = window_sizes
        num_classes = out_channels[0]
        print(num_classes)
        self.pooling = nn.AdaptiveMaxPool2d((1,num_classes))
        
        #grid_size, grid_f_dim
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(
                img_feat_to_grid(self.image_size[i], dims[i], window_sizes[i], dims[i], heads[i])
            )
            
        self.up_projs = nn.ModuleList()
        for i in range(4):
            self.up_projs.append(
                Up(self.out_dim[i], self.out_dim[i+1] // 2, True)
            )
           
        self.ResBlock = nn.ModuleList()
        for i in range(4):
            self.ResBlock.append(
                ResidualBlock(dims[i], dims[i])
            )        
        
    
        self.dim_reducepoint = nn.ModuleList()
        for i in range(4):
            self.dim_reducepoint.append(
                nn.Sequential(
                nn.Linear(dims[i], dims[i]//2),
                nn.LayerNorm(dims[i]//2),
                nn.GELU(),
            )
           )

        self.dim_up = nn.ModuleList()
        for i in range(4):
            self.dim_up.append(
                make_upsample_layers([dims[i],dims[i]])
           )


        cls_head = [256,128,64,64]
   
        final_inp_channels = sum(cls_head)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=1,
                stride=1,
                padding= 0),
            nn.BatchNorm2d(final_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0))        
        
        self.reset_parameters()
    
    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x, id):
        height, width = x.size(2), x.size(3)
        x, fea_list = self.backbone_net(x)
        fea_list = fea_list[::-1]

        x_list = []
        for i in range(4):
            x = self.up_projs[i](x, fea_list[i])
            B,C,H,W = x.size()
            if i<3:
                x = self.stages[i](x)

                x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H//self.window_sizes[i], w=W//self.window_sizes[i])

                x = self.dim_up[i](x)

            x = self.ResBlock[i](x)
            x_list.append(x)

        x1 = F.interpolate(x_list[-1], size=(height, width), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x_list[-2], size=(height, width), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x_list[-3], size=(height, width), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x_list[-4], size=(height, width), mode='bilinear', align_corners=False)        
        x = torch.cat([x1, x2, x3, x4], 1)
        return {'output': torch.sigmoid(self.head(x))}


if __name__ == "__main__":

    model = PointDAT()#.cuda() 
    a = torch.zeros((1, 1, 384, 512))#.cuda()    
    c = model(a,1)    

    