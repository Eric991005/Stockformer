import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class temporalEmbedding(nn.Module):
    def __init__(self, D):
        super(temporalEmbedding, self).__init__()
        self.ff_te = FeedForward([55,D,D])

    def forward(self, TE, T=50):
        '''
        TE:[B,T,2]
        '''
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 5).to(device) # [B,T,5]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(device) # [B,T,50]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 5, 5)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 50, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1) # [B,T,55]
        TE = TE.unsqueeze(dim=2) # [B,T,1,55]
        TE = self.ff_te(TE) # [B,T,1,F]

        return TE  # [B,T,1,F]

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

class sparseSpatialAttention(nn.Module):
    def __init__(self, features, h, d, s):
        super(sparseSpatialAttention, self).__init__()
        self.qfc = nn.Linear(features, features)
        self.kfc = nn.Linear(features, features)
        self.vfc = nn.Linear(features, features)
        self.ofc = nn.Linear(features, features)
        
        self.h = h
        self.d = d
        self.s = s

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(features, features),
                                 nn.ReLU(),
                                 nn.Linear(features, features))

        self.proj = nn.Linear(d, 1)

    def forward(self, x, adjgat):
        '''
        [B,T,N,D]
        '''
        # add spatial positional encoding
        x_ = x + adjgat

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        B, T, N, D = Q.shape

        Q_K = torch.matmul(Q, K.transpose(-2, -1))

        Q_K /= (self.d ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)

        # copy operation
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2,-1)
        value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, N, V.shape[-1])[torch.arange(B)[:, None, None, None],
                                                                                             torch.arange(T)[None, :, None, None],
                                                                                             torch.arange(N)[None, None, :, None],cp,:].squeeze(-2)

        value = self.ofc(value) + x_
        value = self.ln(value)
        return self.ff(value)

class temporalAttention(nn.Module):
    def __init__(self, features, h, d):
        super(temporalAttention, self).__init__()
        self.qfc = FeedForward([features,features])
        self.kfc = FeedForward([features,features])
        self.vfc = FeedForward([features,features])
        self.ofc = FeedForward([features,features])
        self.h = h
        self.d = d
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features,features,features], True)

    def forward(self, x, te, Mask=True):
        '''
        x:[B,T,N,F]
        te:[B,T,N,F]
        '''
        x += te

        query = self.qfc(x).permute(0,2,1,3) #[B,T,N,F]
        key = self.kfc(x).permute(0,2,3,1) #[B,T,N,F]
        value = self.vfc(x).permute(0,2,1,3) #[B,T,N,F]

        attention = torch.matmul(query, key) # [k*B,N,T,T]
        attention /= (self.d ** 0.5) # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attention).to(device) # [k*B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1) # [k*B,N,T,T]

        value = torch.matmul(attention, value).permute(0,2,1,3) # [k*B,N,T,d]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value)

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class temporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(temporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, xh):
        xh = self.tcn(xh.transpose(1,3)).transpose(1,3)
        return xh

class adaptiveFusion(nn.Module):
    def __init__(self, features, h, d):
        super(adaptiveFusion, self).__init__()
        self.qlfc = FeedForward([features,features])
        self.klfc = FeedForward([features,features])
        self.vlfc = FeedForward([features,features])
        self.khfc = FeedForward([features,features])
        self.vhfc = FeedForward([features,features])
        self.ofc = FeedForward([features,features])
        self.h = h
        self.d = d
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features,features,features], True)

    def forward(self, xl, xh, te, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,1,F]
        '''
        xl += te
        xh += te

        query = self.qlfc(xl).permute(0,2,1,3) # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh)).permute(0,2,3,1) # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh)).permute(0,2,1,3) # [B,T,N,F]

        attentionh = torch.matmul(query, keyh) # [k*B,N,T,T]
        
        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attentionh).to(device) # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)
        attentionh /= (self.d ** 0.5) # scaled
        attentionh = F.softmax(attentionh, -1) # [k*B,N,T,T]


        value = torch.matmul(attentionh, valueh).permute(0,2,1,3)
        value = self.ofc(value)
        value = value + xl #+ xh

        value = self.ln(value)

        return self.ff(value)

class dualEncoder(nn.Module):
    def __init__(self, features, h, d, s):
        super(dualEncoder, self).__init__()
        self.tcn = temporalConvNet(features)
        self.tatt = temporalAttention(features, h, d)
        
        self.ssal = sparseSpatialAttention(features, h, d, s)
        self.ssah = sparseSpatialAttention(features, h, d, s)
        
        
    def forward(self, xl, xh, te, adjgat):
        xl = self.tatt(xl, te)
        xh = self.tcn(xh)
        

        spa_statesl = self.ssal(xl,adjgat)
        spa_statesh = self.ssah(xh,adjgat)
        xl = spa_statesl + xl
        xh = spa_statesh + xh
        
        return xl, xh
    
class Stockformer(nn.Module):
    def __init__(self, infea, outfea, L, h, d, s, T1, T2, dev):
        super(Stockformer, self).__init__()
        global device
        device = dev
        self.start_emb_l = FeedForward([infea, outfea, outfea])
        self.start_emb_h = FeedForward([infea, outfea, outfea])
        self.te_emb = temporalEmbedding(outfea)

        self.dual_encoder = nn.ModuleList([dualEncoder(outfea, h, d, s) for i in range(L)])
        self.adaptive_fusion = adaptiveFusion(outfea, h, d)
        
        self.pre_l = nn.Conv2d(T1, T2, (1,1))
        self.pre_h = nn.Conv2d(T1, T2, (1,1))
        
        self.end_emb = FeedForward([outfea, outfea, 1])
        self.end_emb_l = FeedForward([outfea, outfea, 1])

    def forward(self, xl, xh, te, bonus,adjgat):
        '''
        x:[B,T,N]
        bonus:[B,T,N,D2]
        '''
        xl, xh = xl.unsqueeze(-1), xh.unsqueeze(-1) # [B,T,N]->[B,T,N,1]
        xl = torch.concat([xl,bonus],dim = -1) # [B,T,N,1]->[B,T,N,D1+D2]
        xh = torch.concat([xh,bonus],dim = -1) # [B,T,N,1]->[B,T,N,D1+D2]
        xl, xh, TE = self.start_emb_l(xl), self.start_emb_h(xh), self.te_emb(te)

        for enc in self.dual_encoder:
            xl, xh = enc(xl, xh, TE[:,:xl.shape[1],:,:], adjgat)
        
        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)

        hat_y = self.adaptive_fusion(hat_y_l, hat_y_h, TE[:,xl.shape[1]:,:,:])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)
        
        return hat_y.squeeze(-1), hat_y_l.squeeze(-1)
