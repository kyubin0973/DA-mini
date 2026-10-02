"""초기 100사이클로 만드는 피처.

규칙의 근거는 notebooks/01_EDA.ipynb (Q3-2 ~ Q3-5, Q5-1)에 있다.
모든 함수는 수명 값을 쓰지 않으므로 학습 배치와 평가 배치에 똑같이 적용한다.
"""
import numpy as np
import pandas as pd
from scipy.signal import medfilt

from .preprocess import clean_qd, select_cells

K_EARLY, K_LATE = 10, 100        # ΔQ에 쓰는 두 사이클 (측정 순서)
MEDFILT_SIZE = 9                 # 전압 축 중앙값 필터의 크기
SLOPE_FROM, SLOPE_TO = 91, 100   # 기울기를 구하는 측정 순서 구간


def first_valid_index(qdlin):
    """셀별로 Qdlin에 처음 값이 있는 위치(0부터)를 구한다."""
    return pd.Series({cid: int(np.argmax(~np.isnan(arr).all(axis=1)))
                      for cid, arr in qdlin.items()})


def delta_q_features(qdlin, cells, first_valid):
    """분석 대상 셀의 ΔQ(V) 곡선에서 log_var를 계산한다."""
    rows = []
    for cid in cells.loc[cells['use'], 'cell_id']:
        arr, off = qdlin[cid], first_valid[cid]
        dq = arr[off + K_LATE - 1] - arr[off + K_EARLY - 1]
        dq = medfilt(dq.astype(float), MEDFILT_SIZE)
        rows.append({'cell_id': cid, 'dq_var': np.var(dq), 'log_var': np.log10(np.var(dq))})
    return pd.DataFrame(rows)


def qd_slope_features(d_clean, first_valid):
    """정제된 방전 용량에서 91~100번째 측정 사이클의 기울기를 계산한다."""
    e = d_clean.copy()
    e['k'] = e['cycle'] - e['cell_id'].map(first_valid)   # 측정 순서 (1부터)
    e = e[e['k'].between(SLOPE_FROM, SLOPE_TO)]

    rows = []
    for cid, g in e.groupby('cell_id'):
        rows.append({'cell_id': cid,
                     'qd_slope_last': np.polyfit(g['k'], g['QD'], 1)[0],
                     'n_slope': len(g)})
    return pd.DataFrame(rows)


def build_features(df, qdlin):
    """summary와 qdlin에서 분석 대상 셀의 피처 표를 만든다. 셀 선별과 정제를 포함한다."""
    cells = select_cells(df)
    d_clean, _ = clean_qd(df, cells)
    first_valid = first_valid_index(qdlin)

    feat = cells.loc[cells['use'], ['cell_id', 'batch', 'group', 'policy', 'cycle_life']]
    feat = feat.merge(delta_q_features(qdlin, cells, first_valid), on='cell_id')
    feat = feat.merge(qd_slope_features(d_clean, first_valid), on='cell_id')
    feat['log_life'] = np.log10(feat['cycle_life'])
    return feat.reset_index(drop=True)
