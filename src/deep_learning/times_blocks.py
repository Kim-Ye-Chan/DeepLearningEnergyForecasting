import torch
import torch.nn as nn
import torch.nn.functional as F

def fft_for_period(x, k=4):
    # x: (B, T, C)
    xf = torch.fft.rfft(x, dim=1)              # complex
    # 평균 amplitude: (T_freq,)  0(DC)는 제외
    freq_amp = torch.abs(xf).mean(0).mean(-1)  # (T_freq,)
    if freq_amp.numel() > 0:
        freq_amp[0] = 0.0
    top = torch.topk(freq_amp, k=min(k, freq_amp.shape[0]), dim=0).indices
    periods = (x.shape[1] // top.clamp(min=1)).tolist()  # period= T//idx
    # 배치별 가중: (B, k)
    weight = torch.abs(xf).mean(-1)[:, top]              # (B, k)
    return periods, weight  # list[int], (B,k)

class InceptionBlockV1(nn.Module):
    def __init__(self, in_ch, out_ch, num_kernels=6):
        super().__init__()
        ks = [2*i+1 for i in range(num_kernels)]  # 1,3,5,...
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), padding=(0, k//2))
            for k in ks
        ])
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C_in, H, W) 여기선 (B, C, Groups, Period)
        outs = [b(x) for b in self.branches]  # list of (B, C_out, H, W)
        y = torch.stack(outs, dim=-1).mean(-1)  # 평균 앙상블
        return self.act(y)

class TimesBlock(nn.Module):
    """간소화 TimesNet block: [B,T,C] -> [B,T,C]"""
    def __init__(self, d_model, d_ff, num_kernels=6, k_top=4, pred_len=0, seq_len=0):
        super().__init__()
        self.k_top = k_top
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            InceptionBlockV1(in_ch=d_model, out_ch=d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(in_ch=d_ff, out_ch=d_model, num_kernels=num_kernels),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B,T,C)
        B, T, C = x.shape
        periods, pweight = fft_for_period(x, self.k_top)   # periods: len k_top, pweight: (B,k)

        outs = []
        want_len = T + self.pred_len
        for i, period in enumerate(periods):
            period = max(1, int(period))
            # pad to multiple
            if want_len % period != 0:
                new_len = (want_len // period + 1) * period
                pad = x.new_zeros(B, new_len - T, C)
                tmp = torch.cat([x, pad], dim=1)
            else:
                new_len = want_len
                tmp = x

            # (B, new_len//period, period, C) -> conv2d on (C, Groups, Period)
            g = new_len // period
            tmp = tmp.reshape(B, g, period, C).permute(0, 3, 1, 2).contiguous()  # (B,C,g,period)
            y = self.conv(tmp)  # (B,C,g,period)
            y = y.permute(0, 2, 3, 1).reshape(B, -1, C)[:, :want_len, :]         # (B,new_len,C)->cut

            outs.append(y[:, :T, :])  # 현재는 T만 반환

        # 가중합
        if len(outs) == 0:
            return x
        Y = torch.stack(outs, dim=-1)        # (B,T,C,K)
        w = torch.softmax(pweight, dim=1)    # (B,K)
        Y = (Y * w[:, None, None, :]).sum(-1)  # (B,T,C)
        return self.norm(Y + x)
