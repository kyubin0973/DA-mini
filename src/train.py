"""데이터 분할, 교차검증, 평가.

각 선택의 근거와 결과는 notebooks/03_modeling.ipynb에 있다.
저장소 최상위에서 `python -m src.train`으로 실행하면 성능 표를 다시 만든다.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features
from .preprocess import load_cache

SEED = 42
HOLDOUT_EVERY = 4   # 정책 4개마다 하나를 Hold-out으로
HOLDOUT_START = 2   # 평균 수명 순으로 3번째 정책부터 (0부터 세어 2)
N_SPLITS = 5
FINAL_COLS = ['log_var']
FINAL_MODEL = 'Linear'
TARGET = 9.1        # 원 논문의 MAPE (%)


def split_by_policy(data):
    """충전 정책 단위로 Train / Hold-out을 나눈다. 정책을 평균 수명 순으로 정렬해 일정 간격으로 뽑는다."""
    order = data.groupby('policy')['cycle_life'].mean().sort_values().index
    hold_policies = set(order[HOLDOUT_START::HOLDOUT_EVERY])
    is_hold = data['policy'].isin(hold_policies)
    return data[~is_hold].reset_index(drop=True), data[is_hold].reset_index(drop=True)


def mape(y_true_log, y_pred_log):
    """로그 타깃을 원래 단위(사이클)로 되돌린 뒤 MAPE(%)를 계산한다."""
    y, p = 10 ** np.asarray(y_true_log), 10 ** np.asarray(y_pred_log)
    return np.mean(np.abs(p - y) / y) * 100


def make_model(name, alpha=None):
    """표준화와 회귀 모델을 묶은 파이프라인을 만든다."""
    if name == 'Linear':
        reg = LinearRegression()
    elif name == 'Ridge':
        reg = Ridge(alpha=alpha)
    else:
        reg = Lasso(alpha=alpha, max_iter=100000)
    return make_pipeline(StandardScaler(), reg)


def cv_mape(data, cols, name, alpha=None):
    """정책 단위 GroupKFold로 학습 폴드 MAPE, 검증 폴드 MAPE의 평균과 표준편차를 구한다."""
    X, y, g = data[cols].values, data['log_life'].values, data['policy'].values
    tr_scores, va_scores = [], []
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(X, y, g):
        m = make_model(name, alpha).fit(X[tr], y[tr])
        tr_scores.append(mape(y[tr], m.predict(X[tr])))
        va_scores.append(mape(y[va], m.predict(X[va])))
    return np.mean(tr_scores), np.mean(va_scores), np.std(va_scores)


def evaluate(data, model, cols):
    """예측을 붙이고 MAPE, 편향, 편향 제거 후 MAPE를 계산한다."""
    r = data[['cell_id', 'group', 'policy', 'cycle_life', 'log_life']].copy()
    r['pred_log'] = model.predict(data[cols].values)
    r['pred'] = 10 ** r['pred_log']
    r['err_pct'] = (r['pred'] / r['cycle_life'] - 1) * 100       # 양수이면 과대예측
    resid = r['pred_log'] - r['log_life']
    summary = {
        '셀': len(r),
        'MAPE': mape(r['log_life'], r['pred_log']),
        '편향(%)': (10 ** resid.mean() - 1) * 100,
        '편향 제거 후 MAPE': mape(r['log_life'], r['pred_log'] - resid.mean()),
    }
    return r, summary


def run(proc_dir='data/processed', result_dir='results'):
    """피처 생성부터 성능 표 저장까지 실행한다. Batch 1로 학습하고 Batch 2로 평가한다."""
    df, qdlin, _ = load_cache(proc_dir)
    feat = build_features(df, qdlin)
    b1, b2, b3 = (feat[feat['batch'] == b].reset_index(drop=True) for b in (1, 2, 3))

    train, valid = split_by_policy(b1)
    _, cv, _ = cv_mape(train, FINAL_COLS, FINAL_MODEL)
    final = make_model(FINAL_MODEL).fit(train[FINAL_COLS].values, train['log_life'].values)

    v = evaluate(valid, final, FINAL_COLS)[1]
    t = evaluate(b2, final, FINAL_COLS)[1]
    perf = pd.DataFrame([
        {'구분': 'Train (CV)', 'MAPE(%)': cv},
        {'구분': 'Valid (Hold-out)', 'MAPE(%)': v['MAPE']},
        {'구분': 'Test (Batch 2)', 'MAPE(%)': t['MAPE']},
        {'구분': 'Gap (Train - Valid)', 'MAPE(%)': cv - v['MAPE']},
        {'구분': 'Gap (Valid - Test)', 'MAPE(%)': v['MAPE'] - t['MAPE']},
        {'구분': 'Gap (Target - Test)', 'MAPE(%)': TARGET - t['MAPE']},
    ]).round(2)

    by_group = pd.DataFrame({
        'Valid (Batch 1 Hold-out)': v,
        'Batch 2 일반': evaluate(b2[b2['group'] == 'Batch 2 일반'], final, FINAL_COLS)[1],
        'Batch 2 newstructure': evaluate(b2[b2['group'] == 'Batch 2 newstructure'], final, FINAL_COLS)[1],
        'Batch 3': evaluate(b3, final, FINAL_COLS)[1],
    }).T.round(2)

    os.makedirs(result_dir, exist_ok=True)
    perf.to_csv(os.path.join(result_dir, 'model_performance.csv'), index=False)
    by_group.to_csv(os.path.join(result_dir, 'performance_by_group.csv'))
    return perf, by_group


if __name__ == '__main__':
    perf, by_group = run()
    print(perf.to_string(index=False))
    print()
    print(by_group)
