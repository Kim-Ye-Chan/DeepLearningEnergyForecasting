import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if self.mask_flag and attn_mask is not None:
                scores.masked_fill_(attn_mask.mask, -float("inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class _ExoEncoderLayer(nn.Module):
    """ TimeXer의 EncoderLayer 축약: Self-Attn + (글로벌 토큰에만) Cross-Attn + FFN """
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = AttentionLayer(
            FullAttention(mask_flag=False, factor=5, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.cross_attn = AttentionLayer(
            FullAttention(mask_flag=False, factor=5, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff1 = nn.Conv1d(d_model, d_ff, 1)
        self.ff2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, x_tok, cross_seq):
        """
        x_tok:   (B, Ltok, D)   ← exo 토큰열 + 글로벌 토큰
        cross_seq: (B, Lcross, D) ← exo의 원시값 임베딩(시간축 기준), cross attention의 Key/Value
        """
        # Self-Attn on tokens
        x_res = x_tok
        x_tok = self.self_attn(x_tok, x_tok, x_tok, attn_mask=None)[0]
        x_tok = self.dropout(x_tok)
        x_tok = self.norm1(x_res + x_tok)

        # Cross-Attn: 글로벌 토큰 한 개만 cross 쿼리 (마지막 토큰이 글로벌 토큰)
        B, Ltok, D = x_tok.shape
        glb = x_tok[:, -1:, :]                               # (B, 1, D)
        glb_res = glb
        glb = self.cross_attn(glb, cross_seq, cross_seq, attn_mask=None)[0]
        glb = self.dropout(glb)
        glb = self.norm2(glb_res + glb)

        # 글로벌 토큰만 갱신하여 붙이기
        x_tok = torch.cat([x_tok[:, :-1, :], glb], dim=1)    # (B, Ltok, D)

        # FFN
        y = x_tok.transpose(1, 2)                            # (B, D, Ltok)
        y = self.dropout(self.act(self.ff1(y)))
        y = self.dropout(self.ff2(y)).transpose(1, 2)
        x_tok = self.norm3(x_tok + y)
        return x_tok


class ExoPatchEncoder(nn.Module):
    def __init__(
        self,
        c_exo: int,       # 외부변수 채널 수
        d_model: int,
        pred_len: int,
        seq_len: int,     # ★ 추가: 원본 소스 시퀀스 길이
        patch_len: int = 21,
        n_heads: int = 2,
        d_ff: int = 256,
        e_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert c_exo > 0
        self.c_exo = c_exo
        self.d_model = d_model
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.seq_len = seq_len

        self.patch_proj = nn.Linear(patch_len, d_model, bias=False)
        self.pos_emb = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ★ 여기! seq_len을 넣는다
        self.cross_embed = DataEmbedding_inverted(c_in=seq_len, d_model=d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            _ExoEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, activation="gelu")
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x_exo: torch.Tensor):
        B, T, C = x_exo.shape
        assert C == self.c_exo
        # padding to patch
        pad = (self.patch_len - (T % self.patch_len)) % self.patch_len
        if pad:
            x_exo = torch.cat([x_exo, x_exo.new_zeros(B, pad, C)], dim=1)
        Tp = x_exo.shape[1]
        num_patches = Tp // self.patch_len

        # patching per variable
        x = x_exo.permute(0, 2, 1).contiguous().reshape(B * C, num_patches, self.patch_len)
        x = self.patch_proj(x)
        x = x + self.pos_emb(x)
        x = self.dropout(x)

        glb = self.glb_token.expand(B * C, 1, -1)
        x_tok = torch.cat([x, glb], dim=1)  # (B*C, num_patches+1, d_model)

        # cross sequence embedding (TimeXer inverted)
        # DataEmbedding_inverted: (B, T, C) -> (B, C, d_model) using Linear(seq_len->d_model)
        # 주의: forward 내부에서 permute하므로 x_exo shape는 (B, T, C) 그대로 넘김
        cross_seq = self.cross_embed(x_exo[:, :self.seq_len, :], None)  # cross_seq: (B, C, d_model) -> 배리어블 당 하나의 cross key/value 세트
        cross_seq = cross_seq.repeat_interleave(C, dim=0) # x_tok 배치가 (B*C, ... )이므로 batch 축도 동일하게 (B*C, ...)로 확장

        # encoder stack
        for layer in self.layers:
            x_tok = layer(x_tok, cross_seq)

        x_tok = self.norm(x_tok)
        glb_tok = x_tok[:, -1, :]               # (B*C, d_model)
        glb_tok = glb_tok.view(B, C, -1).mean(dim=1)  # (B, d_model)

        delta = self.head(glb_tok)              # (B, pred_len)
        return delta
