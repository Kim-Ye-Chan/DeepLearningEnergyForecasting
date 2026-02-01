import torch
from torch import amp
import lightning as L
from src.deep_learning.quantile_loss import QuantileLoss

class LITransformer(L.LightningModule):
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
        self.kappa          = float(hyperparams_dict.get("kappa", 1.0))
        self.alpha          = float(hyperparams_dict.get("alpha", 0.7))  # NEW
        self.bld_emb   = torch.nn.Embedding(self.num_buildings, self.d_bld)
        self.bld_to_d  = torch.nn.Linear(self.d_bld, self.d_model)
        self.bld_drop  = torch.nn.Dropout(self.emb_dropout)

        assert self.source_length >= 168
        assert self.target_length <= 168

        self.loss = QuantileLoss(quantiles=self.quantiles)
        self.n_quantiles = len(self.quantiles)

        assert self.source_dims is not None and self.target_dims is not None, \
            "Provide source_dims/target_dims for (B, L, D) inputs."
        self.source_project = torch.nn.Linear(self.source_dims, self.d_model)
        self.target_project = torch.nn.Linear(self.target_dims, self.d_model)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

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
        b = self.bld_emb(bld_ids)
        b = self.bld_drop(b)
        b = self.bld_to_d(b).unsqueeze(1)
        return proj_tensor + b

    def _subseq_mask(self, size, device=None):
        device = device or next(self.parameters()).device
        # additive mask: 0(keep) / -inf(block) 형태
        m = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        m = m.masked_fill(m == 1, float('-inf')).masked_fill(m == 0, 0.0)
        return m
    
    def linear_trend(self, past_target):
        # past_target: (B, Ls)
        with amp.autocast(device_type="cuda", enabled=False):
            y = past_target.to(torch.float32).unsqueeze(-1)             # (B, Ls, 1)
            xty = (self.lt_x_t @ y)                                     # (2, Ls) @ (B, Ls, 1) -> (B, 2, 1)
            params = (self.lt_xtx_inv.unsqueeze(0) @ xty).squeeze(-1)   # (1,2,2) @ (B,2,1) -> (B,2)
            slopes, constants = params[:, 0], params[:, 1]              # (B,), (B,)
            future = self.lt_future_idx.unsqueeze(0) * slopes.unsqueeze(-1) + constants.unsqueeze(-1)
        return future.to(past_target.dtype)                              # (B, H)


    def forward(self, source_seq, target_seq, bld_ids=None):
        source = self.dropout(self.source_project(source_seq))
        target = self.dropout(self.target_project(target_seq))
        if bld_ids is not None:
            source = self._add_building_context(source, bld_ids)
            target = self._add_building_context(target, bld_ids)
        tgt_mask = self._subseq_mask(target.size(1), device=target.device)
        transformer_output = self.transformer(
            source, target, tgt_mask=tgt_mask
        )

        with torch.no_grad():
            kt_mag   = (self.kappa * target).abs().mean()
            tout_mag = transformer_output.abs().mean()
            eps = torch.tensor(1e-8, device=kt_mag.device, dtype=kt_mag.dtype)
            self.last_kappa_ratio = (kt_mag / (tout_mag + eps)).detach()
            self.last_kt_mag = kt_mag.detach()
            self.last_tout_mag = tout_mag.detach()

        output = self.output_layer(self.kappa * target + transformer_output)
        return output

    def training_step(self, batch, batch_idx):
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
        preds  = preds[:, -1, :, :]

        residual_true = real_future_target - future_baseline
        eps = 1e-3
        r_std = torch.clamp(residual_true.std(dim=(0,1), keepdim=True, unbiased=False), min=eps)

        residual_true_norm = residual_true / r_std
        preds_slice        = preds[:, self.horizon_start:, :]
        preds_norm         = preds_slice / r_std[..., None]

        loss_main = self.loss.loss(preds_norm, residual_true_norm[:, self.horizon_start:]).mean()

        hs = self.horizon_start
        pred_median_scaled = preds[:, hs:, 1] + future_baseline[:, hs:]
        target_scaled      = real_future_target[:, hs:]
        eps_smape = 1e-3
        smape_loss = (2.0 * torch.abs(pred_median_scaled - target_scaled) /
                      (torch.abs(pred_median_scaled) + torch.abs(target_scaled) + eps_smape)).mean()
        
        loss_total = self.alpha * loss_main + (1.0 - self.alpha) * smape_loss  # NEW

        with torch.no_grad():
            self.log("residual_std_batch", residual_true.std(unbiased=False), on_step=True, prog_bar=True)
        if hasattr(self, "last_kappa_ratio"):
            self.log("kappa_ratio", self.last_kappa_ratio, on_epoch=True, logger=True)
            self.log("kt_mag",      self.last_kt_mag,      on_epoch=True, logger=True)
            self.log("tout_mag",    self.last_tout_mag,    on_epoch=True, logger=True)
        self.log("kappa_value", float(self.kappa), on_epoch=True, logger=True)
        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        preds  = preds[:, -1, :, :]

        residual_true = real_future_target - future_baseline
        eps = 1e-3
        r_std = torch.clamp(residual_true.std(dim=(0,1), keepdim=True, unbiased=False), min=eps)

        residual_true_norm = residual_true / r_std
        preds_slice        = preds[:, self.horizon_start:, :]
        preds_norm         = preds_slice / r_std[..., None]

        loss_main = self.loss.loss(preds_norm, residual_true_norm[:, self.horizon_start:]).mean()

        hs = self.horizon_start
        pred_median_scaled = preds[:, hs:, 1] + future_baseline[:, hs:]
        target_scaled      = real_future_target[:, hs:]
        eps_smape = 1e-3
        smape_loss = (2.0 * torch.abs(pred_median_scaled - target_scaled) /
                      (torch.abs(pred_median_scaled) + torch.abs(target_scaled) + eps_smape)).mean()

        # === NEW: alpha 하이퍼 사용 ===
        loss_total = self.alpha * loss_main + (1.0 - self.alpha) * smape_loss  # NEW

        with torch.no_grad():
            val_bias_scaled = (pred_median_scaled - target_scaled).mean()
            self.log("val_bias_scaled", val_bias_scaled, on_epoch=True, prog_bar=True, logger=True)
            need = (real_future_target - future_baseline)[:, hs:]
            got  = preds[:, hs:, 1]
            self.log("need_mean", need.mean(), on_epoch=True, logger=True)
            self.log("got_mean",  got.mean(),  on_epoch=True, logger=True)
            self.log("residual_std_batch", residual_true.std(unbiased=False), on_step=True, prog_bar=True)
        if hasattr(self, "last_kappa_ratio"):
            self.log("kappa_ratio", self.last_kappa_ratio, on_epoch=True, logger=True)
            self.log("kt_mag",      self.last_kt_mag,      on_epoch=True, logger=True)
            self.log("tout_mag",    self.last_tout_mag,    on_epoch=True, logger=True)
        self.log("kappa_value", float(self.kappa), on_epoch=True, logger=True)
        self.log("val_loss", loss_total, on_epoch=True, prog_bar=True, logger=True)
        return loss_total

    def predict_step(self, batch, batch_idx):
        source_seq, target_seq_covariates, bld_ids = batch
        self._assert_shapes(source_seq, target_seq_covariates, inference=True)
        if self.target_dims is not None:
            expected_cov_D = self.target_dims - 1
            assert target_seq_covariates.size(2) == expected_cov_D, \
                f"infer target covariates D mismatch: {target_seq_covariates.size(2)} != {expected_cov_D}"

        past_target   = source_seq[:, :, -1]
        beta          = self.beta_week  # OVERRIDE 제거됨  # NEW
        week_naive    = past_target[:, self.source_length-168 : self.source_length-168+self.target_length]
        future_linear = self.linear_trend(past_target)
        future_baseline = beta * week_naive + (1.0 - beta) * future_linear

        target_seq_for_decoder = torch.cat((target_seq_covariates, future_baseline.unsqueeze(-1)), dim=2)

        output = self.forward(source_seq, target_seq_for_decoder, bld_ids=bld_ids)
        preds = output.view(output.shape[0], output.shape[1], self.target_length, self.n_quantiles)
        preds = preds[:, -1, :, :]
        return preds, future_baseline

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler}}
