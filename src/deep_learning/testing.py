# Performance testing for Torch deep learning models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Union
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, mean_pinball_loss as pinball, mean_squared_error

# scikit-learn 1.4의 root_mean_squared_log_error 함수를 직접 정의 (호환성 확보)
def rmsle(y_true, y_pred):
    """
    (수정) 음수 값을 처리하기 위해 np.maximum을 사용하여 0 미만의 값을 0으로 만듭니다.
    """
    # ✅ (핵심 수정) np.log1p에 전달하기 전에 모든 값을 0 이상으로 보정합니다.
    y_true_clipped = np.maximum(0, np.array(y_true))
    y_pred_clipped = np.maximum(0, np.array(y_pred))

    y_true_log = np.log1p(y_true_clipped)
    y_pred_log = np.log1p(y_pred_clipped)

    return np.sqrt(np.mean(np.square(y_true_log - y_pred_log)))

def train_val_split(input_sequences: list, output_sequences: list, train_fraction: float = 0.8) -> Tuple[list, list, list, list]:
    """
    (최종 수정) 입력/출력 시퀀스의 길이를 확인하는 안전장치가 추가된,
    데이터 손실 없는 순차 분할 함수입니다.
    """
    # ✅ (핵심 추가) ...const_batch_size의 안전장치를 이식합니다.
    # 입력과 출력 시퀀스의 개수가 다르면 오류를 발생시켜 실수를 방지합니다.
    assert len(input_sequences) == len(output_sequences), "입력과 출력 시퀀스의 개수가 다릅니다!"

    # Get the index of the last training sequence, get training set
    train_end = int(len(input_sequences) * train_fraction)
    train_input_sequences = input_sequences[0:train_end]
    train_output_sequences = output_sequences[0:train_end]

    # Get validation set
    val_input_sequences = input_sequences[train_end:]
    val_output_sequences = output_sequences[train_end:]

    return train_input_sequences, train_output_sequences, val_input_sequences, val_output_sequences
def train_val_split_const_batch_size(
    input_sequences: List[Tuple[int, pd.DataFrame]],
    output_sequences: List[Tuple[int, pd.DataFrame]],
    train_fraction: float = 0.8,
    batch_size: int = 64
) -> Tuple[List[Tuple[int, pd.DataFrame]], List[Tuple[int, pd.DataFrame]], List[Tuple[int, pd.DataFrame]], List[Tuple[int, pd.DataFrame]]]:
    assert len(input_sequences) == len(output_sequences)

    # 1. Split
    train_end = int(len(input_sequences) * train_fraction)
    train_input_sequences = input_sequences[:train_end]
    train_output_sequences = output_sequences[:train_end]
    val_input_sequences = input_sequences[train_end:]
    val_output_sequences = output_sequences[train_end:]

    # 2. Trim training END to align with batch size
    train_remainder = len(train_input_sequences) % batch_size
    if train_remainder > 0:
        train_input_sequences = train_input_sequences[:-train_remainder]
        train_output_sequences = train_output_sequences[:-train_remainder]

    # 3. Trim validation END to align with batch size
    val_remainder = len(val_input_sequences) % batch_size
    if val_remainder > 0:
        val_input_sequences = val_input_sequences[:-val_remainder]
        val_output_sequences = val_output_sequences[:-val_remainder]

    return train_input_sequences, train_output_sequences, val_input_sequences, val_output_sequences


def test_sequences_to_dataframe(test_input_seq, test_output_seq, target_col="전력소비량(kWh)"):
    """
    입력: (id, DataFrame) 튜플 리스트 2개
    출력: time, <target_col>, sequence 컬럼을 가진 DataFrame
    - 타깃은 항상 마지막 열이라는 가정 유지
    - target_col 이름으로 컬럼 생성(하드코딩 제거)
    """
    # Output
    test_output_dates   = np.stack([seq_df.index for _, seq_df in test_output_seq], axis=0)
    test_output_values  = np.stack([seq_df.values for _, seq_df in test_output_seq], axis=0)
    df_test_output = pd.DataFrame({
        "time": np.ravel(test_output_dates),
        target_col: np.ravel(test_output_values[:, :, -1])
    })
    df_test_output["time"] = pd.to_datetime(df_test_output["time"])
    df_test_output["sequence"] = "output"

    # Input
    test_input_dates   = np.stack([seq_df.index for _, seq_df in test_input_seq], axis=0)
    test_input_values  = np.stack([seq_df.values for _, seq_df in test_input_seq], axis=0)
    df_test_input = pd.DataFrame({
        "time": np.ravel(test_input_dates),
        target_col: np.ravel(test_input_values[:, :, -1])
    })
    df_test_input["time"] = pd.to_datetime(df_test_input["time"])
    df_test_input["sequence"] = "input"

    # Concat & sort
    df_test = pd.concat([df_test_output, df_test_input], ignore_index=True)
    df_test = df_test.sort_values("time")
    return df_test


def plot_actual_predicted(df_test, df_preds, model, quantile_interval="95", ax=None, target_col="전력소비량(kWh)"):
    """
    실제값/예측값 전체 구간 플롯.
    target_col을 인자로 받아 사용(하드코딩 제거).
    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.lineplot(
        data=df_test,
        x="time",
        y=target_col,     # <- 하드코딩 제거
        label="Actual values",
        ax=ax
    )

    sns.lineplot(
        data=df_preds,
        x="time",
        y="pred_point",
        label=f"Predictions, {quantile_interval}% quantile interval",
        ax=ax
    )

    ax.fill_between(
        x=df_preds.time,
        y1=df_preds.pred_low,
        y2=df_preds.pred_high,
        label=f"{quantile_interval}% prediction interval",
        color="orange",
        alpha=0.4
    )
    ax.set_title(f"Model: {model}")
    return ax


# testing.py 파일의 plot_sequence_preds 함수를 이걸로 교체해주세요.

def plot_sequence_preds(preds_array, test_input_seq, test_output_seq, model, 
                        target_col="consumption_kWh", sequence_index=0, 
                        quantile_interval="95", ax=None):
    """
    (수정) 튜플 리스트 형태의 데이터를 올바르게 처리하고,
    중복된 컬럼 문제를 해결하여 시각화합니다.
    """
    n_sequences = len(test_output_seq)
    
    # Get predictions for selected sequence
    preds_low = preds_array[sequence_index, :, 0]
    preds_point = preds_array[sequence_index, :, 1]
    preds_high = preds_array[sequence_index, :, -1]

    # 튜플에서 데이터프레임(인덱스 1)을 먼저 선택합니다.
    output_df = test_output_seq[sequence_index][1]
    input_df = test_input_seq[sequence_index][1]

    # ✅ (핵심 수정) 값을 추출하기 전에 중복된 컬럼을 제거합니다.
    output_df = output_df.loc[:, ~output_df.columns.duplicated()]
    input_df = input_df.loc[:, ~input_df.columns.duplicated()]

    # 데이터프레임에서 index와 values를 추출합니다.
    date_output = output_df.index.to_series()
    output = output_df[target_col].values
    
    date_input = input_df.index.to_series()
    input_vals = input_df[target_col].values

    # 이제 input_vals와 output 모두 1D 배열이므로 concatenate가 정상 동작합니다.
    date = pd.concat([date_input, date_output], axis=0)
    actual = np.concatenate([input_vals, output], axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot
    sns.lineplot(x=date, y=actual, label="Actual values", ax=ax)
    sns.lineplot(x=date_output, y=preds_point, label=f"Predictions, {quantile_interval}% quantile interval", ax=ax)
    
    ax.fill_between(
        x=date_output, y1=preds_low, y2=preds_high,
        color="orange", alpha=0.3,
        label=f"Predictions, {quantile_interval}% quantile interval"
    )

    _ = ax.set_title(f"Model: {model},\n Sequence index: {sequence_index} of {n_sequences - 1}")
    _ = ax.set_xlabel("Time")
    _ = ax.set_ylabel("Consumption (kWh)")
    _ = ax.legend()

    return ax

def calculate_metrics(df_actuals, df_preds, model, target_col, quantiles=[.025, 0.5, .975], rounding=4):
    """
    (수정) 데이터 타입을 강제로 숫자로 변환하고 정제하는 기능을 추가하여 안정성을 높였습니다.
    """
    df_actuals_output = df_actuals[df_actuals["sequence"] == "output"].copy()
    
    # ✅ (핵심 수정) 데이터를 사용하기 전에 타입을 숫자로 강제 변환하고, 오류 값은 NaN으로 처리합니다.
    y_true_raw = pd.to_numeric(df_actuals_output[target_col], errors='coerce')
    y_pred_raw = pd.to_numeric(df_preds["pred_point"], errors='coerce')

    # ✅ (핵심 수정) NaN 값을 0으로 채워 계산 오류를 방지합니다.
    y_true = np.nan_to_num(y_true_raw)
    y_pred = np.nan_to_num(y_pred_raw)

    # 길이가 다를 경우를 대비한 안전장치
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    df_metrics = pd.DataFrame([[
        mape(y_true, y_pred) * 100,
        rmsle(y_true, y_pred),
        mae(y_true, y_pred),
        pinball(y_true, np.nan_to_num(pd.to_numeric(df_preds["pred_low"], errors='coerce'))[:min_len], alpha=quantiles[0]),
        pinball(y_true, np.nan_to_num(pd.to_numeric(df_preds["pred_point"], errors='coerce'))[:min_len], alpha=quantiles[1]),
        pinball(y_true, np.nan_to_num(pd.to_numeric(df_preds["pred_high"], errors='coerce'))[:min_len], alpha=quantiles[2])
    ]], 
    columns=[
        "MAPE (%)", "RMSLE", "MAE", 
        f"Pinball Loss ({quantiles[0]:.1%})",
        f"Pinball Loss ({quantiles[1]:.1%})",
        f"Pinball Loss ({quantiles[2]:.1%})"
    ],
    index=[f"Model: {model}"]
    ).round(rounding)

    return df_metrics