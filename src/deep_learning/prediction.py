# Handling & reformatting the predictions of Torch deep learning models
import pandas as pd
import numpy as np

def predictions_to_dataframe(preds_array, original_sequences):
    """
    (수정) 예측 배열과 원본 시퀀스 리스트를 받아 
    안정적으로 결과 데이터프레임을 생성합니다.
    """
    # 1. 예측값에서 필요한 분위수(저점, 중앙값, 고점)를 추출합니다.
    pred_low = preds_array[:, :, 0].flatten()
    pred_point = preds_array[:, :, 1].flatten()
    pred_high = preds_array[:, :, 2].flatten()
    
    # 2. 원본 시퀀스에서 시간 인덱스만 추출하여 하나의 리스트로 만듭니다.
    all_times = []
    for _, seq_df in original_sequences:
        all_times.extend(seq_df.index.tolist())
        
    # 3. 데이터프레임을 생성합니다. (길이가 다르면 오류가 나므로 길이를 맞춰줍니다.)
    min_len = min(len(pred_point), len(all_times))
    
    df = pd.DataFrame({
        'time': all_times[:min_len],
        'pred_low': pred_low[:min_len],
        'pred_point': pred_point[:min_len],
        'pred_high': pred_high[:min_len]
    })
    
    return df