import torch
import lightning as L
import contextlib
from src.deep_learning.quantile_loss import QuantileLoss


# =======================
# Utils & Autocast helper
# =======================
def _get_autocast_dtype():
    # bf16 우선, 없으면 fp16 (GPU 없는 경우 fp32)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # Ampere(8.0+)는 bf16 권장
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


autocast_dtype = _get_autocast_dtype()


# =======================
# TimesNet blocks
# =======================
class Inception_Block_TN(torch.nn.Module):
    """
    - stack/mean 제거 → 학습가능 게이트로 가중합 (einsum)
    - channels_last 지원
    - 필요시 depthwise separable로 교체할 수도 있도록 flag만 남김
    """
    def __init__(self, in_channels, out_channels, num_kernels=6,
                 use_channels_last: bool = False, use_separable: bool = False):
        super().__init__()
        self.use_channels_last = use_channels_last
        self.use_separable = use_separable

        ks = [2 * i + 1 for i in range(num_kernels)]
        if use_separable:
            self.kernels = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1),
                                    padding=(k // 2, 0), groups=in_channels, bias=False),
                    torch.nn.Conv2d(in_channels, in_channels, kernel_size=(1, k),
                                    padding=(0, k // 2), groups=in_channels, bias=False),
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
                )
                for k in ks
            ])
        else:
            self.kernels = torch.nn.ModuleList([
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2)
                for k in ks
            ])

        # 학습 가능한 게이트 (K,) → softmax로 정규화해 가중합
        self.gates = torch.nn.Parameter(torch.ones(len(ks)))

    def forward(self, x):  # x: (B, Cin, P, p)
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        outs = [k(x) for k in self.kernels]               # list of (B,Cout,P,p)
        y = torch.stack(outs, dim=-1)                     # (B,Cout,P,p,K)
        w = torch.softmax(self.gates, dim=0).to(y.dtype)  # (K,)
        return torch.einsum("bcxyk,k->bcxy", y, w)       # (B,Cout,P,p)

@torch.no_grad()
def fft_topk_periods(x, k):
    """
    x: (B, T, C)  ← 가능하면 x는 이미 float32로 전달
    반환: periods(int64, (B,k)), weights(float32, (B,k))
    """
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    xf = torch.fft.rfft(x, dim=1)                 # (B, T//2+1, C)
    amp_bc = xf.abs().mean(-1)                    # (B, T//2+1)
    if amp_bc.size(1) > 0:
        amp_bc[:, 0] = 0                          # DC 제거
    k_eff = min(k, amp_bc.size(1))
    if k_eff == 0:
        B = x.size(0); device = x.device
        return (torch.zeros(B, 0, dtype=torch.long, device=device),
                torch.zeros(B, 0, dtype=torch.float32, device=device))
    topk = torch.topk(amp_bc, k=k_eff, dim=1)
    topk_idx, weights = topk.indices, topk.values
    T = x.size(1)
    periods = (T // topk_idx.clamp_min(1)).to(torch.long)
    return periods, weights


class TimesBlockTN(torch.nn.Module):
    """
    - 주기 버킷 단위로 처리(기존 유니크 주기 루프는 유지하되, 내부 연산을 벡터화/정돈)
    - Inception 블록은 가중합 einsum 사용
    """
    def __init__(self, d_model, d_ff, k_top=2, num_kernels=6, use_channels_last: bool = False,
                 use_separable: bool = False):
        super().__init__()
        self.k = k_top
        self.conv = torch.nn.Sequential(
            Inception_Block_TN(d_model, d_ff, num_kernels=num_kernels,
                               use_channels_last=use_channels_last, use_separable=use_separable),
            torch.nn.GELU(),
            Inception_Block_TN(d_ff, d_model, num_kernels=num_kernels,
                               use_channels_last=use_channels_last, use_separable=use_separable),
        )

    def forward(self, x, precomputed=None):  
        B, T, C = x.shape
        assert precomputed is not None, "precomputed periods/weights required for speed"
        per_tensor, weights = precomputed               
        if per_tensor.numel() == 0:
            return x

        # 가중치 정규화 및 유효성 마스크
        w_norm = torch.softmax(weights, dim=1).to(x.dtype)  
        valid_mask = per_tensor >= 1
        if not valid_mask.any():
            return x

        # 유니크 주기 루프(주기 수는 보통 매우 작음). 각 주기마다 "배치 전체"를 한 번에 처리.
        unique_periods = torch.unique(per_tensor[valid_mask])
        out_acc = torch.zeros_like(x)

        for p in unique_periods.tolist():
            use_mask = (per_tensor == p)                          
            # 샘플별 가중합(동일 샘플 내 k 중 p에 해당하는 항의 합)
            w_p_all = (w_norm * use_mask.to(w_norm.dtype)).sum(dim=1)  
            use_any = w_p_all > 0
            idxs = torch.nonzero(use_any, as_tuple=False).squeeze(1)   

            if idxs.numel() == 0:
                continue

            w_p = w_p_all.index_select(0, idxs)                  
            xb = x.index_select(0, idxs)                         

            # 동일 주기 p에 대해 패딩 1회
            TT = ((T + p - 1) // p) * p
            if TT != T:
                padT = TT - T
                xb = torch.nn.functional.pad(xb, (0, 0, 0, padT))  
            # (Nb,TT,C) -> (Nb,C,P,p)
            # 주의: contiguous 한 번만, 뷰/퍼뮤트 최소화
            P = TT // p
            xx = xb.view(xb.size(0), P, p, C).permute(0, 3, 1, 2).contiguous()  
            yy = self.conv(xx)                                     
            yy = yy.permute(0, 2, 3, 1).reshape(xb.size(0), TT, C)[:, :T, :]

            # 주기별 가중치 적용 후 원배치로 누적
            yy.mul_(w_p.view(-1, 1, 1).to(yy.dtype))               
            out_acc.index_add_(0, idxs, yy)

        return out_acc + x


# =======================
# Model
# =======================
class TimesNet(L.LightningModule):
    """
    LITransformer 인터페이스(배치·베이스라인·로스)는 유지하면서
    백본만 TimesNet(FFT→2D Inception Conv→주기 가중합)으로 교체.
    - 비학습 경로(no_grad) 분리로 Autograd 오버헤드 제거
    - 혼합정밀/채널즈라스트 호환
    - residual scale 파라미터(softplus)로 스파이크 완화 및 수렴 안정
    """
    def __init__(self, hyperparams_dict):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # 필수 하이퍼
        self.source_length  = hyperparams_dict["source_length"]
        self.target_length  = hyperparams_dict["target_length"]
        self.horizon_start  = hyperparams_dict["horizon_start"]
        self.quantiles      = hyperparams_dict["quantiles"]
        self.d_model        = hyperparams_dict["d_model"]
        self.n_encoders     = hyperparams_dict["n_encoders"]
        self.n_decoders     = hyperparams_dict.get("n_decoders", 1)
        self.learning_rate  = hyperparams_dict["learning_rate"]
        self.lr_decay       = hyperparams_dict["lr_decay"]
        self.dropout_rate   = hyperparams_dict["dropout_rate"]
        self.source_dims    = hyperparams_dict.get("source_dims", None)
        self.target_dims    = hyperparams_dict.get("target_dims", None)
        self.beta_week      = hyperparams_dict.get("beta_week", 0.7)
        self.num_buildings  = hyperparams_dict["num_buildings"]
        self.d_bld          = hyperparams_dict.get("d_bld", 16)
        self.emb_dropout    = hyperparams_dict.get("emb_dropout", 0.0)
        self.alpha          = float(hyperparams_dict.get("alpha", 0.7))
        self.use_adaptive_beta = bool(hyperparams_dict.get("use_adaptive_beta", True))

        # TimesNet 전용 하이퍼
        self.top_k        = int(hyperparams_dict.get("top_k", 2))
        self.num_kernels  = int(hyperparams_dict.get("num_kernels", 6))
        self.d_ff_tn      = int(hyperparams_dict.get("d_ff_tn", self.d_model * 2))
        self.use_channels_last = bool(hyperparams_dict.get("use_channels_last", False))
        self.use_separable    = bool(hyperparams_dict.get("use_separable", False))

        # 간단 검증
        assert self.source_length >= 168
        assert self.target_length <= 168
        assert self.source_dims is not None and self.target_dims is not None, \
            "TimesNet을 쓰려면 source_dims/target_dims를 지정하세요."

        # quantiles 정합성 체크
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must contain 0.5 for median-based metrics.")
        self.loss = QuantileLoss(quantiles=self.quantiles)
        self.n_quantiles = len(self.quantiles)
        self.median_idx = self.quantiles.index(0.5)

        # 건물 임베딩
        self.bld_emb   = torch.nn.Embedding(self.num_buildings, self.d_bld)
        self.bld_to_d  = torch.nn.Linear(self.d_bld, self.d_model)
        self.bld_drop  = torch.nn.Dropout(self.emb_dropout)

        # 특성→d_model 프로젝션 (시간=토큰 유지)
        self.source_project = torch.nn.Linear(self.source_dims, self.d_model)  # (B,L,Ds)->(B,L,d)
        self.target_project = torch.nn.Linear(self.target_dims, self.d_model)  # (B,H,Dt)->(B,H,d)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        # TimesNet 백본
        self.encoder = torch.nn.ModuleList([
            TimesBlockTN(self.d_model, self.d_ff_tn, self.top_k, self.num_kernels,
                         use_channels_last=self.use_channels_last, use_separable=self.use_separable)
            for _ in range(self.n_encoders)
        ])
        self.decoder = torch.nn.ModuleList([
            TimesBlockTN(self.d_model, self.d_ff_tn, self.top_k, self.num_kernels,
                         use_channels_last=self.use_channels_last, use_separable=self.use_separable)
            for _ in range(max(1, self.n_decoders))
        ])

        # 출력층: (B,H,d_model) -> (B,H,Q)
        self.output_layer = torch.nn.Linear(self.d_model, self.n_quantiles)

        # Residual scale param (positive via softplus)
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.0))  # softplus(0) ~ 0.693

        # FP32 선형외삽 버퍼 (채널 마지막 상관없음)
        Ls, Hs = self.source_length, self.target_length
        idx  = torch.arange(Ls, dtype=torch.float32)
        ones = torch.ones(Ls, dtype=torch.float32)
        x    = torch.stack((idx, ones), dim=1)
        x_t  = x.transpose(0, 1)
        xtx_inv = torch.linalg.inv(x_t @ x)
        self.register_buffer("lt_x_t", x_t)
        self.register_buffer("lt_xtx_inv", xtx_inv)
        self.register_buffer("lt_future_idx", torch.arange(Ls, Ls+Hs, dtype=torch.float32))

        # torch.compile 시도 (PyTorch 2.1+)
        use_compile = bool(hyperparams_dict.get("use_compile", True))
        if use_compile and torch.cuda.is_available():
            try:
                self.forward = torch.compile(
                    self.forward,
                    mode="max-autotune",
                    fullgraph=False,
                    dynamic=True,
                    options={"cudagraphs": False}   # ★ CUDA Graphs 끔
                )
            except Exception:
                pass

    # ==== utils ====
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

    def _add_building_context(self, x, bld_ids):
        b = self.bld_drop(self.bld_emb(bld_ids))          # (B,d_bld)
        b = self.bld_to_d(b).unsqueeze(1)                 # (B,1,d_model)
        return x + b
    
    def _beta_from_fft(self, periods: torch.Tensor, weights: torch.Tensor, return_strength: bool = False):
        """
        periods, weights: (B,k) from fft_topk_periods (int64, float32)
        use_adaptive_beta=True면 주기 168h 근접도와 파워로 weekly strength를 만들고,
        beta = 0.4 + 0.6 * strength 로 계산. (기존 로직과 동등)
        """
        B = periods.size(0)
        device = periods.device
        dtype_out = torch.float32
        if periods.numel() == 0:
            beta = torch.full((B,), float(self.beta_week), device=device, dtype=dtype_out)
            wk = torch.zeros(B, device=device, dtype=dtype_out)
            top1 = torch.tensor(float("nan"), device=device, dtype=dtype_out)
        else:
            w = torch.softmax(weights.to(dtype_out), dim=1)        # (B,k)
            per_f = periods.to(dtype_out)                          # (B,k)
            target_week = 168.0
            tol = float(getattr(self.hparams, "weekly_tol_hours", 8.0)) if hasattr(self, "hparams") else 8.0

            diff = (per_f - target_week).abs()                     # (B,k)
            idx  = diff.argmin(dim=1)                              # (B,)
            near = diff.gather(1, idx.unsqueeze(1)).squeeze(1) <= tol
            wk   = torch.where(near, w.gather(1, idx.unsqueeze(1)).squeeze(1), torch.zeros_like(w[:, 0]))
            beta = 0.4 + 0.6 * wk if bool(self.use_adaptive_beta) else torch.full((B,), float(self.beta_week), device=device, dtype=dtype_out)
            # 로깅용 대표 주기(배치 첫 샘플의 top-1)
            top1 = per_f[0, idx[0]] if per_f.numel() else torch.tensor(float("nan"), device=device, dtype=dtype_out)

        model_dtype = next(self.parameters()).dtype
        beta = beta.to(model_dtype)
        wk   = wk.to(model_dtype)
        top1 = top1.to(model_dtype)
        return (beta, wk, top1) if return_strength else beta

    def linear_trend(self, past_target):
        y = past_target.to(torch.float32).unsqueeze(-1)        # (B, Ls, 1)
        xty = torch.matmul(self.lt_x_t, y)                     # (2,Ls) @ (B,Ls,1) -> (B,2,1)
        theta = torch.matmul(self.lt_xtx_inv, xty).squeeze(-1) # (B,2)
        slopes, constants = theta[:, 0], theta[:, 1]
        future = self.lt_future_idx.unsqueeze(0) * slopes.unsqueeze(-1) + constants.unsqueeze(-1)
        return future.to(past_target.dtype)                        # (B,H)

    # TimesNet.forward : FFT 호출 금지(외부에서 precomp 제공)
    def forward(self, source_seq, target_seq, bld_ids=None, precomp=None):
        """
        기대 입력: (B, L, Ds), (B, H, Dt)
        precomp: (enc_periods, enc_weights, dec_periods, dec_weights)  # 모두 (B,k)
        """
        src = self.dropout(self.source_project(source_seq))   # (B,L,d)
        tgt = self.dropout(self.target_project(target_seq))   # (B,H,d)

        if bld_ids is not None:
            src = self._add_building_context(src, bld_ids)
            tgt = self._add_building_context(tgt, bld_ids)

        if precomp is None:
            # 안전망(비권장): past_target 기반 간이 계산
            past_target = source_seq[:, :, -1].to(torch.float32).unsqueeze(-1)
            enc_periods, enc_weights = fft_topk_periods(past_target, self.top_k)
            dec_periods, dec_weights = enc_periods, enc_weights
        else:
            enc_periods, enc_weights, dec_periods, dec_weights = precomp

        for blk in self.encoder:
            src = blk(src, precomputed=(enc_periods, enc_weights))     # (B,L,d)

        cond = src.mean(dim=1, keepdim=True)                           # (B,1,d)
        tgt = tgt + cond

        for blk in self.decoder:
            tgt = blk(tgt, precomputed=(dec_periods, dec_weights))     # (B,H,d)

        out = self.output_layer(tgt)                                   # (B,H,Q)
        return out

    # ==== steps ====
    def _baseline_and_periods(self, past_target, target_covariates):
        with torch.no_grad():
            # ① FFT 1회
            past_fp32 = past_target.to(torch.float32).unsqueeze(-1)  # (B,L,1)
            enc_periods, enc_weights = fft_topk_periods(past_fp32, self.top_k)
            precomp = (enc_periods, enc_weights, enc_periods, enc_weights)

            # ② beta를 같은 FFT 결과로 계산
            beta, wk, top1_period = self._beta_from_fft(enc_periods, enc_weights, return_strength=True)

            # ③ 베이스라인 (week-naive & 선형외삽 혼합)
            week_naive     = past_target[:, self.source_length - 168: self.source_length - 168 + self.target_length]
            future_linear  = self.linear_trend(past_target)
            future_baseline = beta.unsqueeze(-1) * week_naive + (1 - beta).unsqueeze(-1) * future_linear

            # ④ 디코더 입력: cov + baseline
            target_seq_for_decoder = torch.cat((target_covariates, future_baseline.unsqueeze(-1)), dim=2)

        return target_seq_for_decoder, future_baseline, precomp, beta, wk, top1_period

    def _compute_loss_bundle(self, preds, future_baseline, real_future_target):
        """
        - residual 표준화(r_std) var_mean로 계산
        - residual_scale(softplus) 적용 (스파이크/오버슈트 완화)
        - QuantileLoss + SMAPE 혼합
        """
        # --- 손실 ---
        with torch.no_grad():
            residual_true = real_future_target - future_baseline
            var, _mean = torch.var_mean(residual_true, dim=1, unbiased=False, keepdim=True)
            r_std = torch.sqrt(var.clamp_min(1e-6))  # (B,1)

        hs = self.horizon_start
        preds_slice = preds[:, hs:, :]  # (B,H-hs,Q)

        # residual scale (positive)
        scale = torch.nn.functional.softplus(self.residual_scale).to(preds_slice.dtype)
        preds_norm = (preds_slice * scale) / r_std[:, :, None]

        residual_true_norm = (real_future_target - future_baseline)[:, hs:] / r_std[:, :, None]

        loss_main = self.loss.loss(preds_norm, residual_true_norm).mean()

        mid = self.median_idx
        pred_median_scaled = preds[:, hs:, mid] + future_baseline[:, hs:]
        target_scaled      = real_future_target[:, hs:]
        eps_smape = 1e-3
        smape_loss = (2.0 * torch.abs(pred_median_scaled - target_scaled) /
                      (torch.abs(pred_median_scaled) + torch.abs(target_scaled) + eps_smape)).mean()

        loss_total = self.alpha * loss_main + (1.0 - self.alpha) * smape_loss
        return loss_total, smape_loss, loss_main, r_std

    def training_step(self, batch, batch_idx):
        source_seq, target_seq, bld_ids = batch
        self._assert_shapes(source_seq, target_seq, inference=False)
        real_future_target = target_seq[:, :, -1]   # (B,H)
        past_target        = source_seq[:, :, -1]   # (B,L)
        target_covariates  = target_seq[:, :, :-1]  # (B,H,Dt-1)

        # --- 베이스라인/FFT (no_grad) ---
        target_seq_for_decoder, future_baseline, precomp, beta, wk, top1_period = \
            self._baseline_and_periods(past_target, target_covariates)

        # --- 모델 전파 (혼합정밀 구간) ---
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
            preds = self.forward(source_seq, target_seq_for_decoder, bld_ids=bld_ids, precomp=precomp)  # (B,H,Q)
            loss_total, smape_loss, loss_main, r_std = \
                self._compute_loss_bundle(preds, future_baseline, real_future_target)

        # --- 로깅 ---
        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_q_loss", loss_main, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_smape", smape_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("beta_mean", beta.mean(), on_epoch=True, prog_bar=False, logger=True)
        self.log("beta_min", beta.min(),  on_epoch=True, prog_bar=False, logger=True)
        self.log("beta_max", beta.max(),  on_epoch=True, prog_bar=False, logger=True)
        if wk is not None:
            self.log("weekly_strength_mean", wk.mean(), on_epoch=True, prog_bar=False, logger=True)
        self.log("top1_period", top1_period, on_epoch=True, prog_bar=False, logger=True)
        # 실단위 bias 추정 참고용
        self.log("residual_std_batch", r_std.mean(), on_epoch=True, prog_bar=False, logger=True)

        return loss_total

    def validation_step(self, batch, batch_idx):
        source_seq, target_seq, bld_ids = batch
        self._assert_shapes(source_seq, target_seq, inference=False)
        real_future_target = target_seq[:, :, -1]
        past_target        = source_seq[:, :, -1]
        target_covariates  = target_seq[:, :, :-1]

        # --- 베이스라인/FFT (no_grad) ---
        target_seq_for_decoder, future_baseline, precomp, beta, wk, top1_period = \
            self._baseline_and_periods(past_target, target_covariates)

        # --- 모델 전파 & 손실 ---
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
            preds = self.forward(source_seq, target_seq_for_decoder, bld_ids=bld_ids, precomp=precomp)  # (B,H,Q)
            loss_total, smape_loss, loss_main, r_std = \
                self._compute_loss_bundle(preds, future_baseline, real_future_target)

        # --- 로깅 ---
        self.log("val_loss", loss_total, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_q_loss", loss_main, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_smape", smape_loss, on_epoch=True, prog_bar=False, logger=True)
        # scaled bias 모니터링
        hs = self.horizon_start
        mid = self.median_idx
        pred_median_scaled = preds[:, hs:, mid] + future_baseline[:, hs:]
        target_scaled      = real_future_target[:, hs:]
        self.log("val_bias_scaled", (pred_median_scaled - target_scaled).mean(),
                 on_epoch=True, prog_bar=True, logger=True)

        if wk is not None:
            self.log("weekly_strength_mean", wk.mean(), on_epoch=True, prog_bar=False, logger=True)
        self.log("beta_mean", beta.mean(), on_epoch=True, prog_bar=False, logger=True)
        self.log("beta_min", beta.min(),  on_epoch=True, prog_bar=False, logger=True)
        self.log("beta_max", beta.max(),  on_epoch=True, prog_bar=False, logger=True)
        self.log("top1_period", top1_period, on_epoch=True, prog_bar=False, logger=True)
        self.log("residual_std_batch", r_std.mean(), on_epoch=True, prog_bar=False, logger=True)

        return loss_total

    def predict_step(self, batch, batch_idx):
        source_seq, target_seq_covariates, bld_ids = batch
        self._assert_shapes(source_seq, target_seq_covariates, inference=True)
        if self.target_dims is not None:
            expected_cov_D = self.target_dims - 1
            assert target_seq_covariates.size(2) == expected_cov_D, \
                f"infer target covariates D mismatch: {target_seq_covariates.size(2)} != {expected_cov_D}"

        past_target = source_seq[:, :, -1]

        with torch.no_grad():
            # FFT 1회
            past_fp32 = past_target.to(torch.float32).unsqueeze(-1)
            enc_periods, enc_weights = fft_topk_periods(past_fp32, self.top_k)
            precomp = (enc_periods, enc_weights, enc_periods, enc_weights)

            # 같은 FFT로 beta 계산 + baseline 생성
            beta = self._beta_from_fft(enc_periods, enc_weights, return_strength=False)  # (B,)
            week_naive    = past_target[:, self.source_length - 168: self.source_length - 168 + self.target_length]
            future_linear = self.linear_trend(past_target)
            future_baseline = beta.unsqueeze(-1) * week_naive + (1 - beta).unsqueeze(-1) * future_linear

            target_seq_for_decoder = torch.cat((target_seq_covariates, future_baseline.unsqueeze(-1)), dim=2)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
            preds = self.forward(source_seq, target_seq_for_decoder, bld_ids=bld_ids, precomp=precomp)  # (B,H,Q)
        return preds, future_baseline
    
    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler}}