import numpy as np
import torch
import lightning as pl

from src.deep_learning.quantile_loss import QuantileLoss

class StatefulQuantileLSTM(pl.LightningModule):
    """StatefulQuantileLSTM
    Stateful LSTM forecasting model, returns quantile predictions.
    Input & output sequences are 3D tensors of shape (batch_size, timesteps, features).
    Hidden & cell states are retained & passed forward across training & inference batches.
    """
    def __init__(self, hyperparams_dict):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.output_length = hyperparams_dict["output_length"]
        self.input_size = hyperparams_dict["input_size"]
        self.horizon_start = hyperparams_dict["horizon_start"]
        self.quantiles = hyperparams_dict["quantiles"]
        self.learning_rate = hyperparams_dict["learning_rate"]
        self.lr_decay = hyperparams_dict["lr_decay"]
        self.num_layers = hyperparams_dict["num_layers"]
        self.hidden_size = hyperparams_dict["hidden_size"]
        self.dropout_rate = hyperparams_dict["dropout_rate"]
        self.feature_indices = hyperparams_dict["feature_indices"]  # 추가

        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.output_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=len(self.quantiles)
        )

        self.loss = QuantileLoss(quantiles=self.quantiles)
        self._median_quantile = np.median(self.quantiles)
        self._median_quantile_idx = self.quantiles.index(self._median_quantile)

        self._last_hiddens_train = None
        self._last_cells_train = None
        self._final_hiddens_train = None
        self._final_cells_train = None

    def forward(self, input_chunk, prev_states=None):
        if prev_states is None:
            lstm_output, (last_hidden_states, last_cell_states) = self.lstm(input_chunk)
        else:
            lstm_output, (last_hidden_states, last_cell_states) = self.lstm(input_chunk, prev_states)
        preds = self.output_layer(lstm_output[:, -1, :])
        return last_hidden_states, last_cell_states, preds

    def backward(self, loss, optimizer=None, optimizer_idx=None):
        if self.trainer.is_last_batch:
            loss.backward(retain_graph=False)
        else:
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        if y is not None:
            y = y.to(self.device) 
        h = 0
        prev_hiddens = []
        prev_cells = []
        batch_preds = []

        input_sequences, output_sequences = batch
        input_seq = input_sequences.clone()  # (batch_size, 72, 31)
        batch_size, seq_len, n_features = input_seq.shape
        idx_c = self.feature_indices["consumption"]  # 전력소비량(kWh) 인덱스
        idx_lag = self.feature_indices["consumption_lag_24h"]
        idx_trend = self.feature_indices["trend"]
        buffer_lag = input_seq[:, -24:, idx_c]  # (batch_size, 24)
        buffer_trend = input_seq[:, -1, idx_trend]  # (batch_size,)

        if self._last_hiddens_train is None:
            last_hidden_states, last_cell_states, preds = self.forward(input_seq)
        else:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states=(self._last_hiddens_train.detach(), self._last_cells_train.detach())
            )

        prev_hiddens.append(last_hidden_states.detach())
        prev_cells.append(last_cell_states.detach())
        batch_preds.append(preds)
        h += 1

        while h < self.output_length:
            pred_c = batch_preds[h - 1][:, self._median_quantile_idx]  # (batch_size,)
            buffer_lag = torch.cat([buffer_lag[:, 1:], pred_c.unsqueeze(1)], dim=1)  # (batch_size, 24)
            lag_val = buffer_lag[:, 0]  # (batch_size,)
            trend_val = buffer_trend + 1.0  # (batch_size,)
            buffer_trend = trend_val

            new_timestep = input_seq[:, -1, :].clone()  # (batch_size, 31)
            new_timestep[:, idx_c] = pred_c
            new_timestep[:, idx_lag] = lag_val
            new_timestep[:, idx_trend] = trend_val

            input_seq = torch.cat((
                input_seq[:, 1:, :],  # (batch_size, 71, 31)
                new_timestep.unsqueeze(1)  # (batch_size, 1, 31)
            ), dim=1)  # (batch_size, 72, 31)

            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states=(prev_hiddens[h-1].detach(), prev_cells[h-1].detach())
            )
            prev_hiddens.append(last_hidden_states.detach())
            prev_cells.append(last_cell_states.detach())
            batch_preds.append(preds)
            h += 1

        preds_horizon = batch_preds[self.horizon_start:]
        preds_horizon = torch.stack(preds_horizon, dim=1)
        targets_horizon = output_sequences[:, self.horizon_start:, 0]
        loss = self.loss.loss(preds_horizon, targets_horizon)

        loss_reduced = loss.mean(dim=2).sum(dim=1).mean()
        self.log("train_loss", loss_reduced, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self._last_hiddens_train = prev_hiddens[-1].detach()
        self._last_cells_train = prev_cells[-1].detach()
        self._final_hiddens_train = prev_hiddens[-1].detach()
        self._final_cells_train = prev_cells[-1].detach()

        return loss_reduced

    def on_train_epoch_end(self):
        self._last_hiddens_train = None
        self._last_cells_train = None

    def reset_states(self):
        self._final_hiddens_train = None
        self._final_cells_train = None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        h = 0
        prev_hiddens = []
        prev_cells = []
        batch_preds = []

        input_sequences, output_sequences = batch
        input_seq = input_sequences.clone()  # (batch_size, 72, 29)
        output_seq = output_sequences[:, 0, 0].unsqueeze(1).clone()  # (batch_size, 1)

        # output_seq를 input_seq의 피처 수(29)에 맞게 확장
        batch_size, seq_len, n_features = input_seq.shape
        idx_c = self.feature_indices["consumption"]  # 예: -1
        idx_lag = self.feature_indices["consumption_lag_24h"]
        idx_trend = self.feature_indices["trend"]
        buffer_lag = input_seq[:, -24:, idx_c]  # (batch_size, 24)
        buffer_trend = input_seq[:, -1, idx_trend]  # (batch_size,)

        if self._final_hiddens_train is None:
            last_hidden_states, last_cell_states, preds = self.forward(input_seq)
        else:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq,
                prev_states=(self._final_hiddens_train, self._final_cells_train)
            )

        prev_hiddens.append(last_hidden_states)
        prev_cells.append(last_cell_states)
        batch_preds.append(preds)
        h += 1

        while h < self.output_length:
            pred_c = batch_preds[h - 1][:, self._median_quantile_idx]  # (batch_size,)
            buffer_lag = torch.cat([buffer_lag[:, 1:], pred_c.unsqueeze(1)], dim=1)  # (batch_size, 24)
            lag_val = buffer_lag[:, 0]  # (batch_size,)
            trend_val = buffer_trend + 1.0  # (batch_size,)
            buffer_trend = trend_val

            new_timestep = input_seq[:, -1, :].clone()  # (batch_size, 29)
            new_timestep[:, idx_c] = pred_c
            new_timestep[:, idx_lag] = lag_val
            new_timestep[:, idx_trend] = trend_val

            input_seq = torch.cat((
                input_seq[:, 1:, :].clone(),  # (batch_size, 71, 29)
                new_timestep.unsqueeze(1)  # (batch_size, 1, 29)
            ), dim=1)  # (batch_size, 72, 29)

            output_seq = output_sequences[:, h, 0].unsqueeze(1).clone()  # (batch_size, 1)

            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq,
                prev_states=(prev_hiddens[h - 1], prev_cells[h - 1])
            )
            prev_hiddens.append(last_hidden_states)
            prev_cells.append(last_cell_states)
            batch_preds.append(preds)
            h += 1

        preds_horizon = batch_preds[self.horizon_start:]
        preds_horizon = torch.stack(preds_horizon, dim=1)
        targets_horizon = output_sequences[:, self.horizon_start:, 0]
        loss = self.loss.loss(preds_horizon, targets_horizon)
        loss_reduced = loss.mean(dim=2).sum(dim=1).mean()

        self.log("val_loss", loss_reduced, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss_reduced

    # lstm.py 파일의 predict_step 함수 전체를 이 코드로 교체해주세요.

    # lstm.py 파일의 predict_step 함수 전체를 이 코드로 교체해주세요.

    def predict_step(self, batch, batch_idx):
        # ✅ (핵심 수정) batch가 리스트이면 첫 번째 요소(입력 텐서)를, 아니면 batch 자체를 사용합니다.
        # 이 한 줄로 val_loader와 test_loader의 경우를 모두 처리할 수 있습니다.
        input_sequences = batch[0] if isinstance(batch, list) else batch

        input_seq = input_sequences.clone()
        h = 0
        prev_hiddens, prev_cells, batch_preds = [], [], []

        idx_c = self.feature_indices["consumption"]
        idx_lag = self.feature_indices["consumption_lag_24h"]
        idx_trend = self.feature_indices["trend"]

        buffer_lag = input_seq[:, -24:, idx_c]
        buffer_trend = input_seq[:, -1, idx_trend]

        if self._final_hiddens_train is None:
            last_hidden_states, last_cell_states, preds = self.forward(input_seq)
        else:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, (self._final_hiddens_train, self._final_cells_train)
            )

        prev_hiddens.append(last_hidden_states)
        prev_cells.append(last_cell_states)
        batch_preds.append(preds)
        h += 1

        while h < self.output_length:
            pred_c = batch_preds[h - 1][:, self._median_quantile_idx]
            buffer_lag = torch.cat([buffer_lag[:, 1:], pred_c.unsqueeze(1)], dim=1)
            lag_val = buffer_lag[:, 0]
            trend_val = buffer_trend + 1.0
            buffer_trend = trend_val

            new_timestep = input_seq[:, -1, :].clone()
            new_timestep[:, idx_c] = pred_c
            new_timestep[:, idx_lag] = lag_val
            new_timestep[:, idx_trend] = trend_val

            input_seq = torch.cat([input_seq[:, 1:, :], new_timestep.unsqueeze(1)], dim=1)

            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, (prev_hiddens[h - 1], prev_cells[h - 1])
            )
            prev_hiddens.append(last_hidden_states)
            prev_cells.append(last_cell_states)
            batch_preds.append(preds)
            h += 1
        
        preds = torch.stack(batch_preds, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler}
        }