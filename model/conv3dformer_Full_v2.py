from importlib import import_module
import torch
import torch.nn as nn
from einops import rearrange
from model_utils.option import args

class GSAttention(nn.Module):
    """global spectral attention (GSA)

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """
    def __init__(self, dim, num_heads):
       
        super(GSAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        #self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_3d = nn.Conv3d(dim, dim*3,kernel_size=1)
        self.gelu = nn.GELU() 

    def forward(self, x):
        b,c,d,h,w = x.shape
        qkv = self.qkv_3d(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        #q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = rearrange(q, 'b (head c) d h w -> b head (c d) (h w)', head=self.num_heads,h=h, w=w, d=d)
        k = rearrange(k, 'b (head c) d h w -> b head (c d) (h w)', head=self.num_heads, h=h, w=w, d=d)
        v = rearrange(v, 'b (head c) d h w -> b head (c d) (h w)', head=self.num_heads, h=h, w=w, d=d)

        #k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, 'b head (c d) (h w) -> b (head c) d h w', head=self.num_heads, h=h, w=w, d=d)
        out = self.gelu(out)

        #out = self.project_out(out)
        return out

    def flops(self,patchresolution):
        flops = 0
        H, W,C = patchresolution
        flops +=  H* C *W* C
        flops +=  C *C*H*W
        return flops




# --------------------------------------------
# Channel Attention (CA) Layer
# --------------------------------------------
class CABlock_3D(nn.Module):
    def __init__(self, c=36, r=15):
        super(CABlock_3D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.fc1=nn.Linear(c, r, bias=False)
        self.relu=nn.LeakyReLU(0.1, inplace=True)
        self.fc2=nn.Linear(r, c, bias=False)

        # self.conv1 = nn.Conv2d(c, c//r, 1, padding=0, bias=False)
        # self.conv2=nn.Conv2d(c//r, c, 1, padding=0, bias=False)

        self.sigmoid=nn.Sigmoid()
        self.weight_GAP = nn.Parameter(torch.Tensor([1.0]))
        self.conv3d_1_1=nn.Conv3d(in_channels = 15,out_channels=36,kernel_size=1, stride=1, padding='same')

        # self.gcb = gap_ch_block

    def forward(self, input_filter, HSI_gap):
        batch, channel, depth, _, _ = input_filter.size()
        se = self.squeeze(input_filter).view(batch, channel*depth)
        se=self.fc1(se)
        se=self.relu(se)
        _, channel_gap = HSI_gap.size()
        # new_HSI_gap = torch.zeros(batch,se.size()[1],1,1)        # pdb.set_trace()
        # pdb.set_trace()

        if args.without_gap == True:
            se=self.fc2(se)
            # se=self.sigmoid(se)
            se=self.sigmoid(se).view(batch, channel, depth, 1,1)

        else:         
            # se=se+(self.weight_GAP*HSI_gap)
            se=se+HSI_gap
            se=self.fc2(se)
            se=self.sigmoid(se).view(batch, channel, depth, 1,1)
            # se=self.sigmoid(se)
                
        # return se
        return input_filter * se.expand_as(input_filter)


class conv3dformer_Full_v2(nn.Module):
    '''RSTCANet'''
    def __init__(self):
        super(conv3dformer_Full_v2, self).__init__()
        # pdb.set_trace()
        self.conv3d_1 = nn.Conv3d(in_channels = 1,out_channels=15,kernel_size=3, stride=2, padding=1)
        #self.conv3d_2 = nn.Conv3d(in_channels = 3,out_channels=3,kernel_size=3)
        self.conv3d_2 = nn.Conv3d(in_channels = 15,out_channels=36,kernel_size=3, stride=2, padding=1)
        self.conv3d_3 = nn.Conv3d(in_channels = 36,out_channels=36,kernel_size=3, stride=(2,2,2), padding=1)
        self.MaxPool3d = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2))

        # self.conv = nn.Conv2d(15, 15, stride=2, kernel_size=3)

        # self.st_CA =CAST.SwinTransformer(
        #         hidden_dim=480,
        #         layers=(2, 2),
        #         heads=(3, 3),
        #         channels=15,
        #         num_classes=36,
        #         head_dim=32,
        #         window_size=7,
        #         downscaling_factors=(4, 4),
        #         relative_pos_embedding=True
        #     )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.final_conv = nn.Conv2d(in_channels=96, out_channels=15, kernel_size=3, stride=1, padding=1, bias=True)
        # self.CA_transformer  = CA_transformer.ViT()

        self.spectral_attention_st1 = GSAttention(dim=15, num_heads=3)
        self.spectral_attention_st2 = GSAttention(dim=36, num_heads=6)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(3528, 1000)
        self.fc1 = nn.Linear(16200, 1000)
        self.fc2 = nn.Linear(1000, 36)
        self.relu=nn.LeakyReLU(0.1)
        self.gelu=nn.GELU()
        
        self.CA_15 = CABlock_3D(c=15*8)
        self.CA_36 = CABlock_3D(c=36*4)
        
    def forward(self, image):
        
        '''
        encoder
        '''
        # pdb.set_trace()
        batch, channel, _, _ = image.size()
        gap = self.avg_pool(image).view(batch, channel) # torch.Size([50, 15])
        gap = gap/gap.max()
        image=image.unsqueeze(dim=1)

        x = self.conv3d_1(image) #input;torch.Size([13, 1, 15, 256, 256]) output; torch.Size([13, 15, 8, 128, 128])
        x = self.relu(x)
        
        shortcut = x
        x = self.CA_15(x,gap)
        x = shortcut+x

        shortcut = x
        x =  self.spectral_attention_st1(x)
        x = shortcut*x
        
        x=self.conv3d_2(x) # output; torch.Size([13, 36, 4, 64, 64])
        x = self.relu(x)
        
        shortcut = x
        x = self.CA_36(x,gap)
        x = shortcut+x

        shortcut = x 
        x =  self.spectral_attention_st2(x)
        x = shortcut*x

        x=self.conv3d_3(x) #output; torch.Size([50, 36, 2, 32, 32])
        x = self.relu(x)

        # pdb.set_trace()
        x= self.MaxPool3d(x) #torch.Size([50, 36, 2, 15, 15])

        # pdb.set_trace()

        x = self.flatten(x)
        # pdb.set_trace()


        # CAB_in = self.conv(image)
        # residual = sf

        x=self.fc1(x)
        x=self.fc2(x)
        # channel_attention = self.CA_transformer(image) #torch.Size([50, 36])

        # cab_st = self.st(CAB_in.permute(0, 3, 1, 2))
        # x_size = (x0.shape[2]//4, x0.shape[3]//4)
        # x = self.m_body(x1, x_size)
        # x = self.m_ipp(x, x_size)
        # x = self.m_conv(x)
        # x = self.m_final_conv(x)
        # x = self.avg_pool(x)

        
        return x
