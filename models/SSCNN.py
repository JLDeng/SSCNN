import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import random
import time
import math

epsilon = 0.0001


class PolynomialRegression(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(PolynomialRegression, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  bias=True)
        self.conv_2 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  bias=True)
        self.conv_3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  bias=True)
        self.conv_4 = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  bias=True)
    
    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        x_3 = self.conv_3(x)
        x_z = self.conv_4(x_1 * x_2) + x_3
        return x_z


class EncoderLayer(nn.Module):
    def __init__(self, d_model, seq_len, pred_len, cycle_len, short_period_len, series_num, kernel_size, long_term=1, short_term=1, seasonal=1, spatial=1, long_term_attn=1, short_term_attn=1, seasonal_attn=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.kernel_size = kernel_size
        self.pred_len = pred_len
        self.series_num = series_num
        self.cycle_len = cycle_len
        self.short_period_len = short_period_len
        self.seq_len = seq_len
        self.long_term = long_term
        self.seasonal = seasonal
        self.short_term = short_term
        self.long_term_attn = long_term_attn
        self.seasonal_attn = seasonal_attn
        self.short_term_attn = short_term_attn
        self.spatial = spatial
        
        self.I_st = Parameter(torch.zeros(1, 1, 1, short_period_len))
        self.E_st = Parameter(torch.zeros(pred_len, 1, 1, short_period_len))
        self.E_si = Parameter(torch.zeros(pred_len, 1, 1, short_period_len))
        self.I_se = Parameter(torch.zeros(seq_len // cycle_len, seq_len // cycle_len, 1, 1))
        
        self.E_se = Parameter(torch.zeros(pred_len // cycle_len + 1, seq_len // cycle_len, 1, 1))
        
        self.skip_conv = nn.Conv2d(in_channels=d_model,
                                  out_channels=d_model,
                                  kernel_size=1,
                                  bias=True)
        self.residual_conv = nn.Conv2d(in_channels=d_model,
                                  out_channels=d_model,
                                  kernel_size=1,
                                  bias=True)
        num_components = long_term + seasonal + short_term + spatial
        self.poly = PolynomialRegression((num_components * 2) * d_model, d_model, kernel_size)

    def forward(self, x):
        b, c, n, t = x.shape
        residual = x
        x_ori = x
        xs = []
        structure_xs = []
        
        ys = []
        x_aux = []
        
        s = []
        hat_s = []
        
        if self.long_term:
        
            mu_lt = (x).mean(-1, keepdim=True).repeat(1, 1, 1, t)
            square_mu_lt = ((x ** 2)).mean(-1, keepdim=True).repeat(1, 1, 1, t)
            var_lt = square_mu_lt - mu_lt ** 2 + epsilon
            r_lt = (x - mu_lt) / (var_lt ** 0.5)
        
            hat_mu_lt = mu_lt[..., -1:].repeat(1, 1, 1, self.pred_len)
            hat_r_lt = r_lt[..., -1:].repeat(1, 1, 1, self.pred_len)
            s.extend([r_lt, mu_lt])
            hat_s.extend([hat_r_lt, hat_mu_lt])
        
        if self.seasonal:
            if self.seasonal_attn:
                I_se = torch.softmax(self.I_se, dim=1)        
                E_se = torch.softmax(self.E_se, dim=1)
            else:
                I_se = torch.softmax(torch.ones(self.I_se.shape).to(self.I_se.device), dim=1)
                E_se = torch.softmax(torch.ones(self.E_se.shape).to(self.E_se.device), dim=1)
            x_cycle = x.reshape(b * c, n, -1, self.cycle_len)
            mu_se = F.conv2d(x_cycle.permute(0, 2, 3, 1), I_se).permute(0, 3, 1, 2).reshape(b, c, n, t)
            square_mu_se = F.conv2d(x_cycle.permute(0, 2, 3, 1) ** 2, I_se).permute(0, 3, 1, 2).reshape(b, c, n, t)
            var_se = square_mu_se - mu_se ** 2 + epsilon
            r_se = (x - mu_se) / (var_se ** 0.5)
        
            hat_mu_se = F.conv2d(mu_se.reshape(b * c, n, -1, self.cycle_len).permute(0, 2, 3, 1), E_se).permute(0, 3, 1, 2).reshape(b, c, n, -1)[..., :self.pred_len]
            hat_r_se = F.conv2d(r_se.reshape(b * c, n, -1, self.cycle_len).permute(0, 2, 3, 1), E_se).permute(0, 3, 1, 2).reshape(b, c, n, -1)[..., :self.pred_len]
            s.extend([r_se, mu_se])
            hat_s.extend([hat_r_se, hat_mu_se])

        if self.short_term:
            if self.short_term_attn:
                I_st = torch.softmax(self.I_st, dim=-1)
                E_st = torch.softmax(self.E_st, dim=-1)
            else:
                I_st = torch.softmax(torch.ones(self.I_st.shape).to(self.I_st.device), dim=1)
                E_st = torch.softmax(torch.ones(self.E_st.shape).to(self.E_st.device), dim=1)
            
            x_pad = F.pad(x, (self.short_period_len - 1, 0), "constant", 0)
            mu_st = F.conv2d(x_pad.reshape(b * c, 1, n, -1), I_st).reshape(b, c, n, t)
            square_mu_st = F.conv2d(x_pad.reshape(b * c, 1, n, -1) ** 2, I_st).reshape(b, c, n, t)
            var_st = square_mu_st - mu_st ** 2 + epsilon
            r_st = (x - mu_st) / (var_st ** 0.5)
        
            hat_mu_st = F.conv2d(mu_st[..., -self.short_period_len:].reshape(b * c, 1, n, -1), E_st).reshape(b, c, -1, n).permute(0, 1, 3, 2)
            hat_r_st = F.conv2d(r_st[..., -self.short_period_len:].reshape(b * c, 1, n, -1), E_st).reshape(b, c, -1, n).permute(0, 1, 3, 2)
            s.extend([r_st, mu_st])
            hat_s.extend([hat_r_st, hat_mu_st])
        
        if self.spatial:
            E_si = torch.softmax(self.E_si, dim=-1)
            I_si = torch.matmul(r_st, r_st.permute(0, 1, 3, 2))
            I_si = (I_si / t).mean(1, keepdim=True)
            I_si = torch.softmax(I_si * 10, dim=-1)
            mu_si = torch.matmul(I_si, r_st)
            var_si = torch.matmul(I_si, r_st ** 2) - mu_si ** 2 + epsilon
            r_si = (r_st - mu_si) / (var_si ** 0.5)
            hat_mu_si = F.conv2d(mu_si[..., -self.short_period_len:].reshape(b * c, 1, n, -1), E_si).reshape(b, c, -1, n).permute(0, 1, 3, 2)
            hat_r_si = F.conv2d(r_si[..., -self.short_period_len:].reshape(b * c, 1, n, -1), E_si).reshape(b, c, -1, n).permute(0, 1, 3, 2)
            s.extend([r_si, mu_si])
            hat_s.extend([hat_r_si, hat_mu_si])
        
        s = torch.cat(s, dim=1)
        hat_s = torch.cat(hat_s, dim=1)
        x = torch.cat([s, hat_s], dim=-1)        
        x = F.pad(x,  mode='constant', pad=(self.kernel_size-1, 0))
        x = self.poly(x)
        x_z = x[...,:-self.pred_len]
        s = x[...,-self.pred_len:]
        x_z = self.residual_conv(x_z)                
        s = self.skip_conv(s)
        return x_z, s


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_layers = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=configs.d_model,
                                    kernel_size=1)
        self.e_layers = configs.e_layers
        for i in range(configs.e_layers):
            self.enc_layers.append(EncoderLayer(configs.d_model, configs.seq_len, configs.pred_len, configs.cycle_len, configs.short_period_len, configs.enc_in, configs.kernel_size, configs.long_term, configs.short_term, configs.seasonal, configs.spatial, configs.long_term_attn, configs.short_term_attn, configs.seasonal_attn))
        self.end_conv = nn.Conv1d(in_channels=configs.d_model * configs.pred_len,
                                  groups=configs.pred_len,
                                  out_channels=configs.pred_len,
                                  kernel_size=1,
                                  bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        input = x_enc.permute(0, 2, 1).unsqueeze(1)
        in_len = input.size(3)
                        
        x = self.start_conv(input)        
        b, c, n, L = x.shape                
        out = 0
        s = 0
        for i in range(self.e_layers):
            start = time.time()
            residual = x
            x, s = self.enc_layers[i](x)
            x = x + residual
            out = s + out
            end = time.time()
        dec_out = self.end_conv(out.permute(0, 3, 1, 2).reshape(b, -1, n))
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                pass
      