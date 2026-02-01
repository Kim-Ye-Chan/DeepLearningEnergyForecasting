import math
from math import sqrt
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
import lightning as L

from src.deep_learning.quantile_loss import QuantileLoss


# -----------------------------
# Embedding & Attention Blocks
# -----------------------------
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, **_):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: (B, T, C)  / x_mark도 (B, T, M)라면 마지막 축 concat
        x = x if x_mark is None else torch.cat([x, x_mark], dim=-1)
        return self.dropout(self.value_embedding(x))  # (B, T, d_model)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -float("inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
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

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
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

        # Cross-Attn: 글로벌 토큰만 cross 쿼리 (마지막 토큰이 글로벌 토큰)
        B, Ltok, D = x_tok.shape
        glb = x_tok[:, -1:, :]
        glb_res = glb
        glb = self.cross_attn(glb, cross_seq, cross_seq, attn_mask=None)[0]
        glb = self.dropout(glb)
        glb = self.norm2(glb_res + glb)

        # 글로벌 토큰만 갱신하여 붙이기
        x_tok = torch.cat([x_tok[:, :-1, :], glb], dim=1)

        # FFN
        y = x_tok.transpose(1, 2)
        y = self.dropout(self.act(self.ff1(y)))
        y = self.dropout(self.ff2(y)).transpose(1, 2)
        x_tok = self.norm3(x_tok + y)
        return x_tok


class ExoPatchEncoder(nn.Module):
    """
    입력:  x_exo (B, T, c_exo)
    출력:  delta (B, pred_len)
    """
    def __init__(
        self,
        c_exo: int,
        d_model: int,
        pred_len: int,
        seq_len: int,     # 호환용(미사용)
        patch_len: int = 21,
        n_heads: int = 2,
        d_ff: int = 256,
        e_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert c_exo > 0 and patch_len > 0
        self.c_exo    = c_exo
        self.d_model  = d_model
        self.pred_len = pred_len
        self.patch_len = patch_len

        self.patch_conv = nn.Conv1d(
            in_channels=c_exo, out_channels=d_model,
            kernel_size=patch_len, stride=patch_len, bias=False
        )
        self.dropout = nn.Dropout(dropout)

        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.layers = nn.ModuleList([
            _ExoEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, activation="gelu")
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_len),
        )
        self.raw_embed = DataEmbedding_inverted(c_in=c_exo, d_model=d_model, dropout=dropout)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        r = T % self.patch_len
        if r != 0:
            pad = self.patch_len - r
            x = torch.cat([x, x.new_zeros(B, pad, C)], dim=1)
        x = x.transpose(1, 2)          # (B, C, T_pad)
        x = self.patch_conv(x)         # (B, d_model, Np)
        x = x.transpose(1, 2)          # (B, Np, d_model)
        return self.dropout(x)

    def forward(self, x_exo: torch.Tensor) -> torch.Tensor:
        tokens = self._patchify(x_exo)                # (B, Np, d_model)
        cross_seq = self.raw_embed(x_exo, None)       # (B, T, d_model)
        B = tokens.size(0)
        glb = self.glb_token.expand(B, 1, -1)
        h = torch.cat([tokens, glb], dim=1)           # (B, Np+1, d_model)
        for layer in self.layers:
            h = layer(h, cross_seq)
        h = self.norm(h)
        glb_out = h[:, -1, :]
        delta = self.head(glb_out)
        return delta


# -----------------------------
# advancedTransformer (patched)
# -----------------------------
class advancedTransformer(L.LightningModule):
    def __init__(self, hyperparams_dict):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # === 기존 하이퍼들 ===
        self.source_length  = hyperparams_dict["source_length"]
        self.target_length  = hyperparams_dict["target_length"]
        self.horizon_start  = hyperparams_dict["horizon_start"]
        self.quantiles      = hyperparams_dict["quantiles"]
        self.d_model        = hyperparams_dict["d_model"]
        self.n_heads        = hyperparams_dict["n_heads"]
        self.n_encoders     = hyperparams_dict["n_encoders"]
        self.n_decoders     = hyperparams_dict["n_decoders"]
        self.d_feedforward  = hyperparams_dict["d_feedforward"]
        self.activation     = hyperparams_dict["activation"]
        self.learning_rate  = hyperparams_dict["learning_rate"]
        self.lr_decay       = hyperparams_dict["lr_decay"]
        self.dropout_rate   = hyperparams_dict["dropout_rate"]
        self.source_dims    = hyperparams_dict.get("source_dims", None)
        self.target_dims    = hyperparams_dict.get("target_dims", None)
        self.beta_week      = hyperparams_dict.get("beta_week", 0.7)
        self.num_buildings  = hyperparams_dict["num_buildings"]
        self.d_bld          = hyperparams_dict.get("d_bld", 16)
        self.emb_dropout    = hyperparams_dict.get("emb_dropout", 0.0)
        self.alpha          = float(hyperparams_dict.get("alpha", 0.7))  # Quantile+sMAPE 가중

        # === RevIN-lite(동적 채널만) 토글 ===
        self.use_revin_dynamic = bool(hyperparams_dict.get("use_revin_dynamic", True))
        self.revin_eps         = float(hyperparams_dict.get("revin_eps", 1e-5))
        dyn_idx = hyperparams_dict.get("dynamic_idx", None)  # 모델 입력 기준 인덱스
        if self.use_revin_dynamic:
            if dyn_idx is None or len(dyn_idx) == 0:
                raise ValueError("use_revin_dynamic=True 인데 dynamic_idx가 비어있습니다.")
            assert self.source_dims is not None and self.target_dims is not None
            if any((int(i) < 0 or int(i) >= self.source_dims) for i in dyn_idx):
                raise ValueError(f"dynamic_idx 범위 오류: 0..{self.source_dims-1}, got {dyn_idx}")
            self.register_buffer("dyn_idx_src", torch.tensor([int(i) for i in dyn_idx], dtype=torch.long), persistent=False)
            dyn_tgt = [int(i) for i in dyn_idx if int(i) < (self.target_dims - 1)]
            if len(dyn_tgt) == 0:
                self.register_buffer("dyn_idx_tgt", torch.empty(0, dtype=torch.long), persistent=False)
            else:
                self.register_buffer("dyn_idx_tgt", torch.tensor(dyn_tgt, dtype=torch.long), persistent=False)
        else:
            self.register_buffer("dyn_idx_src", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("dyn_idx_tgt", torch.empty(0, dtype=torch.long), persistent=False)

        # === 손실/출력 준비 ===
        self.loss = QuantileLoss(quantiles=self.quantiles)
        self.n_quantiles = len(self.quantiles)
        try:
            self.q_med_idx = self.quantiles.index(0.5)
        except ValueError:
            raise ValueError("quantiles 리스트에 0.5(중앙 분위수)가 없습니다.")

        assert self.source_dims is not None and self.target_dims is not None, \
            "Provide source_dims/target_dims for (B, L, D) inputs."
        self.source_project = torch.nn.Linear(self.source_dims, self.d_model)
        self.target_project = torch.nn.Linear(self.target_dims, self.d_model)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        # === 표준 Transformer ===
        self.transformer = torch.nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=self.n_encoders,
            num_decoder_layers=self.n_decoders,
            dim_feedforward=self.d_feedforward,
            dropout=self.dropout_rate,
            activation=self.activation,
            batch_first=True,
        )
        self.temporal_ffn = torch.nn.Sequential(
            torch.nn.Conv1d(self.d_model, self.d_feedforward, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Conv1d(self.d_feedforward, self.d_model, kernel_size=1),
            torch.nn.Dropout(self.dropout_rate),
        )
        self.post_ffn_norm = torch.nn.LayerNorm(self.d_model)
        self.output_layer = torch.nn.Linear(self.d_model, self.target_length * self.n_quantiles)
        assert (self.d_model % self.n_heads) == 0

        # FP32 선형외삽 버퍼
        Ls, Hs = self.source_length, self.target_length
        idx  = torch.arange(Ls, dtype=torch.float32)
        ones = torch.ones(Ls, dtype=torch.float32)
        x    = torch.stack((idx, ones), dim=1)
        x_t  = x.transpose(0, 1)
        xtx_inv = torch.linalg.inv(x_t @ x)
        self.register_buffer("lt_x_t", x_t)
        self.register_buffer("lt_xtx_inv", xtx_inv)
        self.register_buffer("lt_future_idx", torch.arange(Ls, Ls+Hs, dtype=torch.float32))

        # ===== TimeXer (ExoPatch) sidecar =====
        exo_indices = hyperparams_dict.get("exo_indices", None)
        if exo_indices is None or len(exo_indices) == 0:
            raise ValueError("advancedTransformer: 'exo_indices'가 비어있습니다. 동적 외부변수 컬럼을 지정하세요.")
        last_target_idx = self.source_dims - 1  # source_seq의 마지막 채널이 타깃(y)
        if any(int(i) == last_target_idx for i in exo_indices):
            raise ValueError(f"exo_indices에 타깃 채널(index {last_target_idx}) 포함 — 누출 위험")
        if any((int(i) < 0 or int(i) >= self.source_dims) for i in exo_indices):
            raise ValueError(f"exo_indices 범위 오류: [0, {self.source_dims-1}]")

        self.register_buffer("exo_idx", torch.tensor([int(i) for i in exo_indices], dtype=torch.long), persistent=False)
        c_exo = len(exo_indices)

        exo_patch_len = hyperparams_dict.get("exo_patch_len", 21)
        exo_e_layers  = hyperparams_dict.get("exo_e_layers", 1)
        exo_gamma_init = float(hyperparams_dict.get("exo_gamma_init", 0.5))
        exo_dropout = float(hyperparams_dict.get("exo_dropout", 0.10))

        self.sidecar_exo = ExoPatchEncoder(
            c_exo=c_exo,
            d_model=self.d_model,
            pred_len=self.target_length,
            seq_len=self.source_length,
            patch_len=exo_patch_len,
            n_heads=self.n_heads,
            d_ff=self.d_feedforward,
            e_layers=exo_e_layers,
            dropout=exo_dropout,
        )

        # Δ(H) → (H×Q) 확산 (초기엔 중앙만 활성화)
        self.exo_to_q = torch.nn.Linear(1, self.n_quantiles, bias=False)
        with torch.no_grad():
            w = torch.zeros(self.n_quantiles, 1)          # (Q, 1)
            w[self.q_med_idx, 0] = exo_gamma_init
            self.exo_to_q.weight.copy_(w)

        # === 빌딩 임베딩 ===
        self.bld_emb   = torch.nn.Embedding(self.num_buildings, self.d_bld)
        self.bld_to_d  = torch.nn.Linear(self.d_bld, self.d_model)
        self.bld_drop  = torch.nn.Dropout(self.emb_dropout)
        self.use_bld_emb = bool(hyperparams_dict.get("use_bld_emb", True))
        self.bld_scale   = float(hyperparams_dict.get("bld_scale", 1.0))

        # -------------------------
        # [PATCH 1] Residual scale (exp-param)
        # -------------------------
        # 초기 scale=1 정확히 시작: scale = exp(log_res_scale)
        self.log_res_scale = nn.Parameter(torch.zeros(()))

        # -------------------------
        # [PATCH 2] Exo tanh-cap with EMA r_std
        # -------------------------
        # cap = exo_cap_sigma * r_std_ema  (상대 캡)
        self.exo_cap_sigma = float(hyperparams_dict.get("exo_cap_sigma", 2.0))
        self.register_buffer("r_std_ema", torch.tensor(1.0, dtype=torch.float32), persistent=True)
        self.rstd_ema_mom  = float(hyperparams_dict.get("rstd_ema_mom", 0.95))

        # -------------------------
        # [PATCH 3] Quantile order enforcement (권장)
        # -------------------------
        self.enforce_q_order = bool(hyperparams_dict.get("enforce_q_order", True))

    # ---------- 내부 유틸 ----------
    def _assert_shapes(self, source_seq, target_seq, *, inference: bool = False):
        assert source_seq.ndim == 3 and target_seq.ndim == 3
        B, L, Ds = source_seq.shape
        B2, H, Dt = target_seq.shape
        assert L == self.source_length
        assert H == self.target_length
        if self.source_dims is not None:
            assert Ds == self.source_dims
        if self.target_dims is not None and not inference:
            assert Dt == self.target_dims

    def _add_building_context(self, proj_tensor, bld_ids):
        if not self.use_bld_emb or self.bld_scale == 0.0:
            return proj_tensor
        b = self.bld_emb(bld_ids)
        b = self.bld_drop(b)
        b = self.bld_to_d(b).unsqueeze(1)
        return proj_tensor + self.bld_scale * b

    def _subseq_mask(self, size, device=None):
        device = device or next(self.parameters()).device
        m = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        m = m.masked_fill(m == 1, float('-inf')).masked_fill(m == 0, 0.0)
        return m

    @torch.no_grad()
    def _revin_dynamic_(self, x: torch.Tensor, idx: torch.Tensor):
        """
        In-place RevIN-lite for selected dynamic channels.
        x:  (B, L, D)
        idx:(K,)  dynamic channel indices (model input 기준)
        """
        if idx.numel() == 0:
            return x
        x_sel = torch.index_select(x, dim=2, index=idx)        # (B, L, K)
        mean  = x_sel.mean(dim=1, keepdim=True)                # (B, 1, K)
        std   = x_sel.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.revin_eps)
        x_sel = (x_sel - mean) / std
        idx_exp = idx.view(1, 1, -1).expand(x.size(0), x.size(1), -1)
        x.scatter_(dim=2, index=idx_exp, src=x_sel)
        return x

    def linear_trend(self, past_target):
        with amp.autocast(device_type="cuda", enabled=False):
            y = past_target.to(torch.float32).unsqueeze(-1)             # (B, Ls, 1)
            xty = (self.lt_x_t @ y)                                     # (2, Ls) @ (B, Ls, 1) -> (B, 2, 1)
            params = (self.lt_xtx_inv.unsqueeze(0) @ xty).squeeze(-1)   # (1,2,2) @ (B,2,1) -> (B,2)
            slopes, constants = params[:, 0], params[:, 1]
            future = self.lt_future_idx.unsqueeze(0) * slopes.unsqueeze(-1) + constants.unsqueeze(-1)
        return future.to(past_target.dtype)                             # (B, H)

    # [PATCH 1] 잔차 전체 배율 적용 함수
    def _apply_res_scale(self, x):
        scale = torch.exp(self.log_res_scale) + 1e-6  # 완전 0 방지
        return x * scale

    # (권장) 분위수 순서 강제
    def _apply_q_order(self, preds):
        if not self.enforce_q_order or preds.size(-1) < 3:
            return preds
        low  = preds[..., 0]
        med  = torch.maximum(preds[..., 1], low + 1e-6)
        high = torch.maximum(preds[..., 2], med + 1e-6)
        return torch.stack([low, med, high], dim=-1)

    @torch.no_grad()
    def _update_rstd_ema(self, r_std_batch_scalar: torch.Tensor):
        # r_std_batch_scalar: 1-element tensor or scalar
        mom = self.rstd_ema_mom
        val = float(r_std_batch_scalar)
        self.r_std_ema.mul_(mom).add_((1.0 - mom) * val)

    # ---------- 순전파 ----------
    def forward(self, source_seq, target_seq, bld_ids=None):
        # RevIN-lite(동적만)
        if self.use_revin_dynamic:
            source_seq = self._revin_dynamic_(source_seq.clone(), self.dyn_idx_src)
            target_seq = self._revin_dynamic_(target_seq.clone(), self.dyn_idx_tgt)

        source = self.dropout(self.source_project(source_seq))   # (B,Ls,D)->(B,Ls,d_model)
        target = self.dropout(self.target_project(target_seq))   # (B,Lt,D)->(B,Lt,d_model)
        if bld_ids is not None:
            source = self._add_building_context(source, bld_ids)
            target = self._add_building_context(target, bld_ids)

        tgt_mask = self._subseq_mask(target.size(1), device=target.device)
        transformer_output = self.transformer(source, target, tgt_mask=tgt_mask)  # (B,Lt,d_model)

        y = transformer_output + self.temporal_ffn(transformer_output.transpose(1, 2)).transpose(1, 2)
        y = self.post_ffn_norm(y)

        output = self.output_layer(y)  # (B, Lt, H*Q)
        return output

    # ---------- 학습/검증/예측 ----------
    def training_step(self, batch, batch_idx):
        source_seq, target_seq, bld_ids = batch
        self._assert_shapes(source_seq, target_seq, inferenZce=False)
        real_future_target = target_seq[:, :, -1]      # (B,H)
        past_target        = source_seq[:, :, -1]      # (B,Ls)
        target_covariates  = target_seq[:, :, :-1]     # (B,H,Dt-1)

        week_naive     = past_target[:, self.source_length-168 : self.source_length-168+self.target_length]
        future_linear  = self.linear_trend(past_target)
        future_baseline = self.beta_week * week_naive + (1.0 - self.beta_week) * future_linear  # (B,H)

        target_seq_for_decoder = torch.cat((target_covariates, future_baseline.unsqueeze(-1)), dim=2)
        output = self.forward(source_seq, target_seq_for_decoder, bld_ids=bld_ids)   # (B,Lt,H*Q)
        preds  = output.view(output.shape[0], output.shape[1], self.target_length, self.n_quantiles)
        preds  = preds[:, -1, :, :]  # (B, H, Q)

        # ----- ExoPatch 경로 -----
        x_exo = source_seq.index_select(dim=2, index=self.exo_idx)   
        delta = self.sidecar_exo(x_exo)                             
        exo_q = self.exo_to_q(delta.unsqueeze(-1))                   

        residual_true_all = real_future_target - future_baseline
        r_std_batch = residual_true_all.std(unbiased=False).clamp_min(1e-3)
        self._update_rstd_ema(r_std_batch)
        exo_cap = self.exo_cap_sigma * self.r_std_ema 
        exo_q = torch.tanh(exo_q / exo_cap) * exo_cap

        preds = preds + exo_q

        # [PATCH 1] 잔차 전체 배율(learnable) 적용 (+ 분위수 정렬)
        preds = self._apply_res_scale(preds)
        preds = self._apply_q_order(preds)

        # ----- 손실 -----
        eps = 1e-3
        # 기존 손실은 호라이즌별/배치 std로 노멀라이즈
        r_std = torch.clamp(residual_true_all.std(dim=(0,1), keepdim=True, unbiased=False), min=eps)
        residual_true_norm = residual_true_all / r_std
        preds_slice        = preds[:, self.horizon_start:, :]
        preds_norm         = preds_slice / r_std[..., None]

        loss_main = self.loss.loss(preds_norm, residual_true_norm[:, self.horizon_start:]).mean()

        hs = self.horizon_start
        pred_median_scaled = preds[:, hs:, self.q_med_idx] + future_baseline[:, hs:]
        target_scaled      = real_future_target[:, hs:]
        eps_smape = 1e-3
        smape_loss = (2.0 * torch.abs(pred_median_scaled - target_scaled) /
                      (torch.abs(pred_median_scaled) + torch.abs(target_scaled) + eps_smape)).mean()

        loss_total = self.alpha * loss_main + (1.0 - self.alpha) * smape_loss

        # 로깅
        with torch.no_grad():
            pred_med_real = preds[:, hs:, self.q_med_idx] + future_baseline[:, hs:]
            target_real   = real_future_target[:, hs:]
            val_bias_unscaled = (pred_med_real - target_real).mean()
            self.log("val_bias_unscaled", val_bias_unscaled, on_epoch=True, logger=True)
            self.log("r_std_ema", self.r_std_ema, on_epoch=True, logger=True)

            exo_delta_mean = delta.mean()
            exo_delta_std  = delta.std(unbiased=False)
            self.log("exo_delta_mean", exo_delta_mean, on_epoch=True, logger=True)
            self.log("exo_delta_std",  exo_delta_std,  on_epoch=True, logger=True)

            beta = self.beta_week
            beta_week_eff = beta * week_naive[:, hs:].mean() + (1.0 - beta) * future_linear[:, hs:].mean()
            self.log("beta_week_eff", beta_week_eff, on_epoch=True, logger=True)

        return loss_total

    def validation_step(self, batch, batch_idx):
        source_seq, target_seq, bld_ids = batch
        self._assert_shapes(source_seq, target_seq, inference=False)
        real_future_target = target_seq[:, :, -1]
        past_target        = source_seq[:, :, -1]
        target_covariates  = target_seq[:, :, :-1]

        week_naive     = past_target[:, self.source_length-168 : self.source_length-168+self.target_length]
        future_linear  = self.linear_trend(past_target)
        future_baseline = self.beta_week * week_naive + (1.0 - self.beta_week) * future_linear

        target_seq_for_decoder = torch.cat((target_covariates, future_baseline.unsqueeze(-1)), dim=2)
        output = self.forward(source_seq, target_seq_for_decoder, bld_ids=bld_ids)
        preds  = output.view(output.shape[0], output.shape[1], self.target_length, self.n_quantiles)
        preds  = preds[:, -1, :, :]  # (B, H, Q)

        # ----- ExoPatch 경로 -----
        x_exo = source_seq.index_select(dim=2, index=self.exo_idx)
        delta = self.sidecar_exo(x_exo)                      # (B, H)
        exo_q = self.exo_to_q(delta.unsqueeze(-1))           # (B, H, Q)

        # [PATCH 2] tanh-cap (상대 캡: exo_cap_sigma * r_std_ema)
        residual_true_all = real_future_target - future_baseline
        r_std_batch = residual_true_all.std(unbiased=False).clamp_min(1e-3)
        self._update_rstd_ema(r_std_batch)  # 검증에서도 업데이트(드리프트 추정 안정)
        exo_cap = self.exo_cap_sigma * self.r_std_ema
        exo_q = torch.tanh(exo_q / exo_cap) * exo_cap

        preds = preds + exo_q

        # [PATCH 1] 잔차 전체 배율 + 분위수 정렬
        preds = self._apply_res_scale(preds)
        preds = self._apply_q_order(preds)

        eps = 1e-3
        r_std = torch.clamp(residual_true_all.std(dim=(0,1), keepdim=True, unbiased=False), min=eps)
        residual_true_norm = residual_true_all / r_std
        preds_slice        = preds[:, self.horizon_start:, :]
        preds_norm         = preds_slice / r_std[..., None]

        loss_main = self.loss.loss(preds_norm, residual_true_norm[:, self.horizon_start:]).mean()

        hs = self.horizon_start
        pred_median_scaled = preds[:, hs:, self.q_med_idx] + future_baseline[:, hs:]
        target_scaled      = real_future_target[:, hs:]
        eps_smape = 1e-3
        smape_loss = (2.0 * torch.abs(pred_median_scaled - target_scaled) /
                      (torch.abs(pred_median_scaled) + torch.abs(target_scaled) + eps_smape)).mean()

        loss_total = self.alpha * loss_main + (1.0 - self.alpha) * smape_loss

        # 로깅
        with torch.no_grad():
            pred_med_real = preds[:, hs:, self.q_med_idx] + future_baseline[:, hs:]
            target_real   = real_future_target[:, hs:]
            val_bias_unscaled = (pred_med_real - target_real).mean()
            self.log("val_bias_unscaled", val_bias_unscaled, on_epoch=True, logger=True)
            self.log("r_std_ema", self.r_std_ema, on_epoch=True, logger=True)

            exo_delta_mean = delta.mean()
            exo_delta_std  = delta.std(unbiased=False)
            self.log("exo_delta_mean", exo_delta_mean, on_epoch=True, logger=True)
            self.log("exo_delta_std",  exo_delta_std,  on_epoch=True, logger=True)

            beta = self.beta_week
            beta_week_eff = beta * week_naive[:, hs:].mean() + (1.0 - beta) * future_linear[:, hs:].mean()
            self.log("beta_week_eff", beta_week_eff, on_epoch=True, logger=True)

        self.log("val_loss", loss_total, on_epoch=True, prog_bar=True, logger=True)
        self.log("exo_gamma_med",
                 float(self.exo_to_q.weight[self.q_med_idx, 0].detach().cpu()),
                 on_epoch=True, logger=True)
        return loss_total

    def predict_step(self, batch, batch_idx):
        source_seq, target_seq_covariates, bld_ids = batch
        self._assert_shapes(source_seq, target_seq_covariates, inference=True)
        if self.target_dims is not None:
            expected_cov_D = self.target_dims - 1
            assert target_seq_covariates.size(2) == expected_cov_D, \
                f"infer target covariates D mismatch: {target_seq_covariates.size(2)} != {expected_cov_D}"

        past_target   = source_seq[:, :, -1]
        beta          = self.beta_week
        week_naive    = past_target[:, self.source_length-168 : self.source_length-168+self.target_length]
        future_linear = self.linear_trend(past_target)
        future_baseline = beta * week_naive + (1.0 - beta) * future_linear

        target_seq_for_decoder = torch.cat((target_seq_covariates, future_baseline.unsqueeze(-1)), dim=2)
        output = self.forward(source_seq, target_seq_for_decoder, bld_ids=bld_ids)
        preds = output.view(output.shape[0], output.shape[1], self.target_length, self.n_quantiles)
        preds = preds[:, -1, :, :]  # (B, H, Q)

        # ----- ExoPatch 경로 -----
        x_exo = source_seq.index_select(dim=2, index=self.exo_idx)
        delta = self.sidecar_exo(x_exo)                      # (B, H)
        exo_q = self.exo_to_q(delta.unsqueeze(-1))           # (B, H, Q)

        # [PATCH 2] tanh-cap (상대 캡: exo_cap_sigma * r_std_ema)
        exo_cap = self.exo_cap_sigma * self.r_std_ema
        exo_q = torch.tanh(exo_q / exo_cap) * exo_cap

        preds = preds + exo_q

        # [PATCH 1] 잔차 전체 배율 + 분위수 정렬
        preds = self._apply_res_scale(preds)
        preds = self._apply_q_order(preds)

        # ★ 예측 안전을 위해 FP32 고정
        preds = preds.to(torch.float32).contiguous()
        future_baseline = future_baseline.to(torch.float32).contiguous()

        return preds, future_baseline

    def configure_optimizers(self):
        base_lr = float(self.learning_rate)
        gamma   = float(self.lr_decay)

        exo_lr_mult = float(getattr(self, "hparams", {}).get("exo_lr_mult", 2.0))  # ★ 권장: 1.5~2.0

        exo_params      = list(self.exo_to_q.parameters())
        sidecar_params  = list(self.sidecar_exo.parameters())

        exo_ids     = {id(p) for p in exo_params}
        sidecar_ids = {id(p) for p in sidecar_params}
        banned_ids  = exo_ids | sidecar_ids

        base_params = [p for p in self.parameters() if id(p) not in banned_ids]

        optimizer = torch.optim.Adam(
            [
                {"params": base_params,     "lr": base_lr},
                {"params": sidecar_params,  "lr": base_lr * exo_lr_mult},
                {"params": exo_params,      "lr": base_lr * exo_lr_mult},
            ],
            lr=base_lr,
            weight_decay=1e-4,   # ★ 약한 정규화 권장
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler}}
