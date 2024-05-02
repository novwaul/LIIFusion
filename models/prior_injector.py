import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorInjector(nn.Module):
    def __init__(self, n_feats=64, n_stl = 6):
        super().__init__()
        self.inner = nn.Sequential(*[STL(i, n_feats) for i in range(n_stl)])
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def _padWithReflect(self, img):
        *_, H, W = img.shape
        padding_bottom = 8-H%8
        padding_right = 8-W%8
        return F.pad(img, (0, padding_right, 0, padding_bottom), mode='reflect')
    
    def _crop(self, img, shape):
        *_, H, W = shape
        return img[:,:,:(H),:(W)]
    
    def forward(self, feat, prior): 
        shape = feat.shape
        raw = self._padWithReflect(feat)
        cvrt_img = raw.permute(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
        cvrt_img, _ = self.inner((cvrt_img, prior))
        cvrt_img = cvrt_img.permute(0, 3, 1, 2)# (B,H,W,C) -> (B,C,H,W)
        out = self.conv(cvrt_img)
        return self._crop(raw + out, shape)

class STL(nn.Module):
    def __init__(self, order, n_feats):
        super().__init__()
        cycShft = (order%2 != 0)
        self.n_feats = n_feats
        self.ln1 = nn.LayerNorm(n_feats)
        self.inner1 = MSA(cycShft, n_feats)

        self.inner2 = nn.Sequential(
            nn.LayerNorm(n_feats),
            MLP(n_feats)
        )
        self.kernel = nn.Sequential(
             nn.Linear(4*n_feats, 2*n_feats, bias=False),
             nn.GELU()
         )

    def forward(self, inp):
        cvrt_img, prior = inp

        mean_var = self.kernel(prior).view(-1, 1, 1, self.n_feats*2) # shape: (B, 64) -> (B, 1, 1, 128)
        p_u, p_s = mean_var.chunk(2, dim=3) # (B, 1, 1, 128)-> 2 x (B, 1, 1, 64)
        
        x = self.ln1(cvrt_img)
        p = x*p_s + p_u  
        
        x = self.inner1(x, p)
        z = x + cvrt_img

        r = z
        z = self.inner2(z)
        out = z + r

        return (out, prior)

class MSA(nn.Module):
    def __init__(self, cycShft, n_feats):
        super().__init__()
        self.cyc_shft_wndw_partition = CycShftWndwPartition(8, cycShft)
        self.self_attention = SelfAttention(n_feats)
        self.un_cyc_shft_wndw_partition = UnCycShftWndwPartition(8, cycShft)
    
    def forward(self, cvrt_img, prior):
        windows, prior, mask, shape = self.cyc_shft_wndw_partition(cvrt_img, prior)
        windows = self.self_attention(windows, prior, mask)
        new_cvrt_img = self.un_cyc_shft_wndw_partition(windows, shape)
        return new_cvrt_img #(B, H, W, C)

class CycShftWndwPartition(nn.Module):
    def __init__(self, window_size, cycShft):
        super().__init__()
        self.wsize = window_size
        self.cycShft = cycShft
        self.h_slices = [slice(0,-window_size), slice(-window_size,-window_size//2), slice(-window_size//2,None)]
        self.w_slices = [slice(0,-window_size), slice(-window_size,-window_size//2), slice(-window_size//2,None)]
    
    def _mask(self, H, W):
        att_partition = torch.zeros((1,H,W))
        attention_idx = 0
        for h_slice in self.h_slices:
            for w_slice in self.w_slices:
                att_partition[:,h_slice,w_slice] = attention_idx
                attention_idx += 1
        att_partition = att_partition.view(1, H//self.wsize, self.wsize, W//self.wsize, self.wsize)
        att_partition = att_partition.transpose(2, 3).reshape(-1, self.wsize*self.wsize)
        mask = att_partition.unsqueeze(1) - att_partition.unsqueeze(2) #(i,j): 0 if "i" is in same window with "j"
        mask = mask.masked_fill(mask==0, 0.0)
        mask = mask.masked_fill(mask!=0, -100.0)
        return mask # (H/w*W/w, N, N)

    def forward(self, cvrt_img, prior):
        B, H, W, C = cvrt_img.shape
        if self.cycShft:
            x = torch.roll(cvrt_img, shifts=(-8//2,-8//2), dims=(1,2))
            p = torch.roll(prior, shifts=(-8//2,-8//2), dims=(1,2))
            mask = self._mask(H,W).to(x.device)
        else:
            x = cvrt_img
            p = prior
            mask = torch.zeros((H*W//(self.wsize*self.wsize),self.wsize*self.wsize,self.wsize*self.wsize)).to(x.device)
        x = x.view(B, H//self.wsize, self.wsize, W//self.wsize, self.wsize, C)
        p = p.view(B, H//self.wsize, self.wsize, W//self.wsize, self.wsize, C)
        windows = x.transpose(2, 3).reshape(-1, self.wsize*self.wsize, C) #(B=B*H/w*W/w, N=w*w, C)
        prior = p.transpose(2, 3).reshape(-1, self.wsize*self.wsize, C)

        return windows, prior, mask, (B, H, W, C)

class UnCycShftWndwPartition(nn.Module):
    def __init__(self, window_size, cycShft):
        super().__init__()
        self.wsize = window_size
        self.cycShft = cycShft
    
    def forward(self, windows, shape):
        B, H, W, C = shape
        x = windows.view(B, H//self.wsize, W//self.wsize, self.wsize, self.wsize, C)
        x = x.transpose(2, 3).reshape(B, H, W, C)
        if self.cycShft:
            cvrt_img = torch.roll(x, shifts=(8//2,8//2), dims=(1,2))
        else:
            cvrt_img = x
        return cvrt_img

class SelfAttention(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.kv = nn.Linear(n_feats, 2*n_feats, bias=False)
        self.q = nn.Linear(n_feats, n_feats, bias=False)
        self.biasMatrix = nn.Parameter(torch.zeros((2*8-1)**2, 4))
        self.relativeIndex = self._getRelativeIndex()
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(n_feats, n_feats)

    def _getRelativeIndex(self):
        h_cord = torch.arange(8)
        w_cord = torch.arange(8)
        h_grid, w_grid = torch.meshgrid([h_cord, w_cord]) # (8,8), (8,8)
        x = torch.stack((h_grid, w_grid)) # (2,8,8)
        x = torch.flatten(x, 1) # (2,64)
        x = x.unsqueeze(dim=2) - x.unsqueeze(dim=1) #(2,64,64), (i,j): distance from i to j
        x[0,:,:] += (8-1)
        x[0,:,:] *= (2*8 - 1)
        x[1,:,:] += (8-1)
        relative_index_matrix = x[0,:,:] + x[1,:,:] # (64,64)
        return relative_index_matrix.reshape(-1)

    def forward(self, windows, prior, mask):
        B, N, C = windows.shape
        WNum, *_ = mask.shape #(windownum, N, N)

        kv = self.kv(windows).view(B, N, 2, 4, C//4).permute(2,0,3,1,4) #(3,B,headnum,N,dimension)
        k,v = kv[0], kv[1]

        q = self.q(prior).view(B, N, 1, 4, C//4).permute(2,0,3,1,4)
        q = q[0]

        x = torch.matmul(q, k.transpose(-2,-1)) / ((C//4)**0.5) #(B,headnum,N,N)
        relative_pos_bias = self.biasMatrix[self.relativeIndex].view((8*8),(8*8),4).permute(2,0,1) #(headnum,64,64)
        x = x+relative_pos_bias.unsqueeze(dim=0) #(B,headnum,N=w*w=64,N)
        x = x.view(B//WNum, WNum, 4, N, N).transpose(1, 2) + mask.view(1, 1, WNum, N, N)
        x = x.transpose(1,2).reshape(-1, 4, N, N)
        attention = self.softmax(x)
        self_attention = torch.matmul(attention, v) #(B,headnum,N,dimension)
        z = self_attention.transpose(1,2).reshape(B, N, C)
        new_windows = self.proj(z)
        return new_windows #(B, w*w, C)

class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Linear(n_feats,2*n_feats),
            nn.GELU(),
            nn.Linear(2*n_feats,n_feats)
        )
    
    def forward(self, cvrt_img):
        return self.inner(cvrt_img)