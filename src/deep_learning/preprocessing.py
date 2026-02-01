# Preprocessing steps for the Torch deep learning models
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple
from typing import Union
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_transformer_sequences(df: pd.DataFrame, input_seq_length: int = 72, output_seq_length: int = 33, forecast_t: int = 15) -> Tuple[list, list]:
    """
    Takes in the consumption training dataset.
    Returns it as a pair of lists: Input & output sequences, each sequence a dataframe.
    No shifting or lagging, in contrast to the sequencing done in the analysis part. 
    T = first forecast hour in output sequence.
    """

    # Find the index of the first row in the data at hour T, where the index is bigger than input_seq_length. This will be the first forecast point.
    # EXAMPLE: input_seq_length = 72, first T index = 72, [0, 71] = 72 input steps.
    first_t = df.loc[(df.date.dt.hour == forecast_t) & (df.index >= input_seq_length)].index.values[0]

    # Find the index of the last row in the data at hour T, with `output_seq_length` time steps after it. This will be the last forecast point.
    # EXEAMPLE: output_seq_length = 33, last T index = 72, [72, 104] = 33 output steps.
    last_t = df.loc[(df.date.dt.hour == forecast_t) & (df.index + output_seq_length - 1 <= df.index.values[-1])].index.values[-1]

    # Number of T rows followed by a sufficient length input & output sequence
    n_sequences = (last_t - first_t) // 24 + 1 

    # Initialize lists of sequences
    input_sequences = []
    output_sequences = []
    
    # Get sequences
    for t in range(first_t, last_t + 1, 24):

        # Get input sequence [t-72, t)
        new_input = pd.concat([
            df.iloc[(t - input_seq_length):t, 0], # Time
            df.iloc[(t - input_seq_length):t, 1], # Past target
            df.iloc[(t - input_seq_length):t, 2:] # Past covariates
            ], axis = 1)
        new_input = new_input.set_index("date")
    
        # Get output sequence [t, t+H) 
        new_output = pd.concat([
            df.iloc[t:(t + output_seq_length), 0], # Time 
            df.iloc[t:(t + output_seq_length), 1], # Future target
            df.iloc[t:(t + output_seq_length), 2:] # Future known covariates
            ], axis = 1)
        new_output = new_output.set_index("date")
    
        # Concatenate to arrays of sequences
        input_sequences.append(new_input)
        output_sequences.append(new_output)

    return input_sequences, output_sequences

from sklearn.preprocessing import StandardScaler, MinMaxScaler  # (호환용; 직접 객체는 사용하지 않음)
import numpy as np
import pandas as pd

class SequenceScaler:
    def __init__(self, 
                 target_col: str = "전력소비량(kWh)", 
                 standard_scale_cols: list = None,
                 minmax_scale_cols: list = None,
                 passthrough_cols: list = None,
                 # === 새 옵션(최소 수정) ===
                 per_building: bool = False,           # 건물별 타깃 정규화 사용 여부
                 groupby_key: str = None,              # 그룹핑 키(예: '건물번호')
                 target_minmax: tuple = (0.0, 1.0),    # 타깃 스케일 범위
                 use_group_for: tuple = ("target",)):  # ('target',) 이면 타깃만 그룹별 스케일링
        """
        기존 기능은 유지하면서, 건물별(Target) MinMax(0–1) 정규화를 선택적으로 지원.
        """
        self.target_col = target_col
        self.standard_scale_cols = standard_scale_cols or []
        self.minmax_scale_cols = minmax_scale_cols or []
        self.passthrough_cols = passthrough_cols or []

        # TARGET이 다른 피처 리스트에 중복 포함되지 않도록
        self.feature_cols = self.standard_scale_cols + self.minmax_scale_cols + self.passthrough_cols
        if self.target_col in self.feature_cols:
            self.feature_cols.remove(self.target_col)

        # 새 옵션 저장
        self.per_building = per_building
        self.groupby_key = groupby_key
        self.target_min, self.target_max = target_minmax
        self.use_group_for = tuple(use_group_for) if use_group_for is not None else tuple()

        # 통계 저장소
        self.stats = {}         # 전역 통계(기존)
        self.group_stats = {}   # 그룹별 통계(신규): {group_id: {'target': {'min':..., 'max':...}}}

    def fit(self, input_df: list, output_df: list) -> None:
        """
        각 피처 그룹에 맞는 통계치(평균/표준편차 또는 최소/최대)를 계산하고 저장.
        - per_building=True 이고 'target'을 그룹 스케일 대상으로 설정한 경우:
            타깃은 '건물별 Min/Max'로 저장(self.group_stats)
          그 외:
            기존대로 전역 mean/std 또는 min/max 저장(self.stats)
        """
        # 인덱스 리셋 + 중복 컬럼 제거
        all_dfs = [seq_df.reset_index(drop=True) for _, seq_df in input_df] + \
                  [seq_df.reset_index(drop=True) for _, seq_df in output_df]
        for i, df in enumerate(all_dfs):
            all_dfs[i] = df.loc[:, ~df.columns.duplicated()]

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # StandardScaler용 통계치(전역)
        for col in self.standard_scale_cols:
            if col not in combined_df.columns: 
                continue
            col_series = combined_df[col].astype(float)
            std = float(col_series.std(ddof=0))
            if std > 0 and np.isfinite(std):
                self.stats[col] = {
                    'mean': float(col_series.mean()),
                    'std':  std
                }

        # MinMaxScaler용 통계치(전역, -1~1 맵핑 용도)
        for col in self.minmax_scale_cols:
            if col not in combined_df.columns:
                continue
            col_series = combined_df[col].astype(float)
            cmin, cmax = float(col_series.min()), float(col_series.max())
            if np.isfinite(cmin) and np.isfinite(cmax) and (cmax - cmin) > 0:
                self.stats[col] = {'min': cmin, 'max': cmax}

        # Target 통계
        if self.target_col in combined_df.columns:
            if self.per_building and (self.groupby_key is not None) and ("target" in self.use_group_for):
                # 건물별 Min/Max 저장
                if self.groupby_key not in combined_df.columns:
                    raise KeyError(f"groupby_key='{self.groupby_key}' 컬럼이 데이터에 없습니다.")
                grp = combined_df[[self.groupby_key, self.target_col]].copy()
                grp[self.target_col] = grp[self.target_col].astype(float)
                gstats = grp.groupby(self.groupby_key)[self.target_col].agg(['min', 'max']).reset_index()
                self.group_stats = {}
                for _, row in gstats.iterrows():
                    g = row[self.groupby_key]
                    mn, mx = float(row['min']), float(row['max'])
                    if np.isfinite(mn) and np.isfinite(mx) and (mx - mn) > 0:
                        self.group_stats[g] = {'target': {'min': mn, 'max': mx}}
                # 폴백을 위해 전역 통계도 하나 저장(필요 시)
                # 전역 표준화/역변환은 사용하지 않지만, 안전용으로 mean/std 저장
                t_series = combined_df[self.target_col].astype(float)
                t_std = float(t_series.std(ddof=0))
                self.stats[self.target_col] = {
                    'mean': float(t_series.mean()),
                    'std':  (t_std if (t_std > 0 and np.isfinite(t_std)) else 1.0)
                }
            else:
                # 기존 전역 표준화(mean/std)
                t_series = combined_df[self.target_col].astype(float)
                t_std = float(t_series.std(ddof=0))
                if t_std > 0 and np.isfinite(t_std):
                    self.stats[self.target_col] = {
                        'mean': float(t_series.mean()),
                        'std':  t_std
                    }
                else:
                    self.stats[self.target_col] = {
                        'mean': float(t_series.mean()),
                        'std':  1.0
                    }

    def transform(self, scale_df: list) -> list:
        """
        fit에서 계산된 통계치를 사용하여 각 컬럼에 맞는 스케일링을 적용.
        - 타깃:
          * per_building=True & groupby_key 사용 & 'target' 그룹 스케일 대상이면
            → 그룹별 MinMax(target_min, target_max)로 변환
          * 그 외 → 기존 전역 표준화(mean/std)
        """
        transformed = []
        for bid, seq_df in scale_df:
            # 중복 컬럼 제거
            seq_df_unique = seq_df.loc[:, ~seq_df.columns.duplicated()]
            scaled_seq = seq_df_unique.copy()

            # StandardScaler (전역)
            for col in self.standard_scale_cols:
                if col in self.stats and 'std' in self.stats[col] and col in seq_df_unique.columns:
                    m, s = self.stats[col]['mean'], self.stats[col]['std']
                    s = 1.0 if (s is None or s == 0 or not np.isfinite(s)) else s
                    scaled_seq[col] = (seq_df_unique[col].astype(float) - m) / s

            # MinMaxScaler (전역, -1 ~ 1)
            for col in self.minmax_scale_cols:
                if col in self.stats and ('min' in self.stats[col]) and ('max' in self.stats[col]) and col in seq_df_unique.columns:
                    mn, mx = self.stats[col]['min'], self.stats[col]['max']
                    denom = (mx - mn) if (mx is not None and mn is not None and (mx - mn) != 0) else 1.0
                    scaled_seq[col] = -1 + 2 * (seq_df_unique[col].astype(float) - mn) / denom

            # Target 스케일링
            if (self.target_col in scaled_seq.columns):
                if self.per_building and (self.groupby_key is not None) and ("target" in self.use_group_for):
                    # 시퀀스는 단일 건물로 구성되어 있어야 함
                    if self.groupby_key not in scaled_seq.columns:
                        raise KeyError(f"groupby_key='{self.groupby_key}' 컬럼이 시퀀스에 없습니다.")
                    # 그룹 id 추출(단일 값 가정)
                    gvals = scaled_seq[self.groupby_key].unique()
                    if len(gvals) != 1:
                        # 그래도 첫 값을 사용하고 경고 없이 진행(최소 수정)
                        g = gvals[0]
                    else:
                        g = gvals[0]
                    if g in self.group_stats and 'target' in self.group_stats[g]:
                        mn = self.group_stats[g]['target']['min']
                        mx = self.group_stats[g]['target']['max']
                        denom = (mx - mn) if (mx - mn) != 0 else 1.0
                        a, b = self.target_min, self.target_max
                        # MinMax to [a,b]
                        scaled_seq[self.target_col] = a + (scaled_seq[self.target_col].astype(float) - mn) * (b - a) / denom
                    else:
                        # 그룹 통계가 없으면 전역 표준화로 폴백
                        m, s = self.stats[self.target_col]['mean'], self.stats[self.target_col]['std']
                        s = 1.0 if (s is None or s == 0 or not np.isfinite(s)) else s
                        scaled_seq[self.target_col] = (scaled_seq[self.target_col].astype(float) - m) / s
                else:
                    # 기존 전역 표준화
                    if self.target_col in self.stats and 'std' in self.stats[self.target_col]:
                        m, s = self.stats[self.target_col]['mean'], self.stats[self.target_col]['std']
                        s = 1.0 if (s is None or s == 0 or not np.isfinite(s)) else s
                        scaled_seq[self.target_col] = (scaled_seq[self.target_col].astype(float) - m) / s

            # 최종 배열: 원래 DF 컬럼 순서 유지(타깃은 맨 뒤)
            cols_wo_target = [c for c in seq_df_unique.columns if c != self.target_col]
            final_cols = cols_wo_target + ([self.target_col] if self.target_col in seq_df_unique.columns else [])
            transformed.append((bid, scaled_seq[final_cols].values.astype(np.float32)))

        return transformed

    def backtransform_preds(self, preds_array: np.ndarray, group_ids: list = None) -> np.ndarray:
        """
        예측값 역변환.
        - per_building=True & group_ids 제공:
            그룹별 MinMax 역변환: y = (y' - a) * (mx - mn)/(b - a) + mn
        - 그 외:
            전역 표준화 역변환: y = z * std + mean
        preds_array shape: (N, H, Q) 또는 (N, H)
        group_ids: 길이 N의 리스트/배열 (각 시퀀스의 그룹 id, 예: 건물번호)
        """
        # 배치 축 정규화
        arr = np.asarray(preds_array)
        if arr.ndim == 2:   # (N,H) → (N,H,1)로 통일
            arr = arr[..., None]

        N = arr.shape[0]

        if self.per_building and (self.groupby_key is not None) and ("target" in self.use_group_for) and (group_ids is not None):
            if len(group_ids) != N:
                raise ValueError(f"group_ids 길이({len(group_ids)})가 예측 N({N})과 다릅니다.")
            a, b = self.target_min, self.target_max
            out = np.empty_like(arr, dtype=np.float32)
            for i in range(N):
                g = group_ids[i]
                gs = self.group_stats.get(g, {}).get('target', None)
                if gs is None:
                    # 그룹 통계가 없으면 전역 표준화 폴백
                    m = self.stats[self.target_col]['mean']; s = self.stats[self.target_col]['std']
                    s = 1.0 if (s is None or s == 0 or not np.isfinite(s)) else s
                    out[i] = arr[i] * s + m
                else:
                    mn, mx = gs['min'], gs['max']
                    denom = (b - a) if (b - a) != 0 else 1.0
                    out[i] = (arr[i] - a) * (mx - mn) / denom + mn
            return out if preds_array.ndim == 3 else out[..., 0]

        # 전역 표준화 역변환(기존)
        if self.target_col not in self.stats or 'mean' not in self.stats[self.target_col] or 'std' not in self.stats[self.target_col]:
            return preds_array  # 통계 없으면 그대로 반환
        m = self.stats[self.target_col]['mean']
        s = self.stats[self.target_col]['std']
        s = 1.0 if (s is None or s == 0 or not np.isfinite(s)) else s
        out = arr * s + m
        return out if preds_array.ndim == 3 else out[..., 0]

    
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_seq: np.ndarray, output_seq: Optional[np.ndarray] = None):
        self.input_seq = torch.tensor(input_seq, dtype=torch.float32)
        self.output_seq = (
            torch.tensor(output_seq, dtype=torch.float32) if output_seq is not None else None
        )

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        if self.output_seq is None:
            return self.input_seq[idx]
        return self.input_seq[idx], self.output_seq[idx]

class SequenceDatasetWithBld(torch.utils.data.Dataset):
    """
    기존 SequenceDataset의 규약을 유지:
      - output_seq가 None이면 __getitem__은 (input,) 형식이었지만
        여기서는 모델이 bld_ids를 필요로 하므로 (input, bld_ids) 반환
      - output_seq가 있으면 (input, output, bld_ids) 반환

    기대 shape:
      input_seq : (N, L, D_in)
      output_seq: (N, H, D_out)  또는 None
      bld_ids   : (N,)  (int64)
    """
    def __init__(
        self,
        input_seq: np.ndarray,
        output_seq: Optional[np.ndarray] = None,
        bld_ids: Optional[np.ndarray] = None,
    ):
        assert bld_ids is not None, "bld_ids를 반드시 전달하세요. shape=(N,)"
        # 길이 일치 확인
        N = len(input_seq)
        assert len(bld_ids) == N, f"bld_ids 길이({len(bld_ids)}) != input_seq 길이({N})"
        if output_seq is not None:
            assert len(output_seq) == N, f"output_seq 길이({len(output_seq)}) != input_seq 길이({N})"

        # 텐서 변환
        self.input_seq  = torch.tensor(input_seq,  dtype=torch.float32)
        self.output_seq = torch.tensor(output_seq, dtype=torch.float32) if output_seq is not None else None
        # 건물 ID는 long (Embedding용)
        self.bld_ids    = torch.tensor(bld_ids,    dtype=torch.long)

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        x = self.input_seq[idx]
        b = self.bld_ids[idx]
        if self.output_seq is None:
            # 예: 인퍼런스에서 target covariates를 따로 안쓰는 경우
            return x, b
        y = self.output_seq[idx]
        # 학습/검증, 또는 예측에서 covariates-only 타깃을 쓸 때
        return x, y, b

