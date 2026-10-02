"""데이터 로딩, 셀 선별, 방전 용량 정제.

규칙과 기준값의 근거는 notebooks/01_EDA.ipynb (Q2-1 ~ Q2-5, Q3-1)에 있다.
"""
import os

import numpy as np
import pandas as pd

FILES = {
    1: '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    2: '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    3: '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}

N_KEEP = 101      # 셀마다 저장할 Qdlin 사이클 수
N_POINTS = 1000   # Qdlin 한 곡선의 포인트 수

QD_MIN, QD_MAX = 0.8, 1.15   # 방전 용량의 정상 범위 (Ah)
EOL_CHECK = 0.885            # 마지막 용량이 이 값 이하이면 EOL 도달
DEV_WINDOW = 11              # 이동 중앙값을 구할 사이클 수
DEV_MAX = 0.04               # 이동 중앙값과의 차이 허용 한계 (Ah)


def load_mat(path):
    """.mat 파일 로더. v7.3(HDF5)은 mat73, 그 이하는 scipy로 읽는다."""
    import mat73
    import scipy.io as sio
    try:
        data = mat73.loadmat(path)
    except Exception:
        data = sio.loadmat(path, simplify_cells=True)
    return data


def to_list_of_dicts(d):
    """mat73의 dict-of-lists 구조를 list-of-dicts로 변환한다."""
    if isinstance(d, dict):
        keys = list(d.keys())
        n = len(d[keys[0]])
        return [{k: d[k][i] for k in keys} for i in range(n)]
    return d


def extract_summary(batch, batch_no):
    """배치 내 모든 셀의 summary를 사이클 단위 DataFrame으로 변환한다."""
    records = []
    for i, cell in enumerate(batch):
        s = cell['summary']
        qd = np.array(s['QDischarge'])
        qc = np.array(s['QCharge'])
        ir = np.array(s['IR'])
        tmax = np.array(s['Tmax'])
        tavg = np.array(s['Tavg'])
        tmin = np.array(s['Tmin'])
        ct = np.array(s['chargetime'])

        cycle_life = float(cell['cycle_life'])
        policy = str(cell.get('policy_readable') or cell.get('policy') or 'unknown')

        for c in range(len(qd)):
            records.append({
                'batch': batch_no,
                'cell_id': f'b{batch_no}c{i}',
                'cycle': c + 1,
                'cycle_life': cycle_life,
                'charging_policy': policy,
                'QD': qd[c],
                'QC': qc[c],
                'IR': ir[c],
                'Tmax': tmax[c],
                'Tavg': tavg[c],
                'Tmin': tmin[c],
                'chargetime': ct[c],
            })
    return pd.DataFrame(records)


def extract_qdlin(batch, batch_no):
    """배치 내 모든 셀의 초기 N_KEEP개 사이클 Qdlin을 (N_KEEP, N_POINTS) 배열로 모은다."""
    qdlin = {}
    for i, cell in enumerate(batch):
        q = cell['cycles']['Qdlin']
        arr = np.full((N_KEEP, N_POINTS), np.nan, dtype=np.float32)
        for j in range(min(N_KEEP, len(q))):
            if q[j] is not None and np.size(q[j]) == N_POINTS:
                arr[j] = q[j]
        qdlin[f'b{batch_no}c{i}'] = arr
    return qdlin


def build_cache(data_dir, proc_dir):
    """.mat 원본에서 summary.parquet과 qdlin.npz를 만든다. 최초 1회만 실행한다."""
    os.makedirs(proc_dir, exist_ok=True)
    dfs, qdlin, vdlin = [], {}, None
    for b, fname in FILES.items():
        print(f'Batch {b} 로딩 중...')
        mat = load_mat(os.path.join(data_dir, fname))
        if vdlin is None:
            vdlin = np.array(mat['batch']['Vdlin'][0])
        batch = to_list_of_dicts(mat['batch'])
        dfs.append(extract_summary(batch, b))
        qdlin.update(extract_qdlin(batch, b))
        del mat, batch

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(os.path.join(proc_dir, 'summary.parquet'))
    np.savez_compressed(os.path.join(proc_dir, 'qdlin.npz'), Vdlin=vdlin, **qdlin)
    return df, qdlin, vdlin


def load_cache(proc_dir):
    """build_cache가 저장한 summary, qdlin, vdlin을 불러온다."""
    df = pd.read_parquet(os.path.join(proc_dir, 'summary.parquet'))
    z = np.load(os.path.join(proc_dir, 'qdlin.npz'))
    vdlin = z['Vdlin']
    qdlin = {k: z[k] for k in z.files if k != 'Vdlin'}
    return df, qdlin, vdlin


def select_cells(df):
    """셀 단위 표를 만들고 분석 대상 여부(use)를 붙인다."""
    cells = (df.drop_duplicates('cell_id')
             [['batch', 'cell_id', 'cycle_life', 'charging_policy']]
             .reset_index(drop=True))

    ok = df[df['QD'].between(QD_MIN, QD_MAX)]
    last = (ok.groupby('cell_id')
            .agg(n_cycles=('cycle', 'max'), qd_last=('QD', 'last'))
            .reset_index())
    cells = cells.merge(last, on='cell_id')

    cells['reached_eol'] = cells['qd_last'] <= EOL_CHECK
    cells['use'] = cells['reached_eol'] & cells['cycle_life'].notna()

    # 집단 구분: Batch 2는 newstructure 여부로 나눈다
    ns = cells['charging_policy'].str.contains('newstructure')
    cells['group'] = 'Batch ' + cells['batch'].astype(str)
    cells.loc[(cells['batch'] == 2) & ns, 'group'] = 'Batch 2 newstructure'
    cells.loc[(cells['batch'] == 2) & ~ns, 'group'] = 'Batch 2 일반'
    cells['policy'] = cells['charging_policy'].str.replace('-newstructure', '', regex=False)
    return cells


def clean_qd(df, cells):
    """분석 대상 셀의 사이클별 기록에서 방전 용량 이상값을 제거한다."""
    use_ids = cells.loc[cells['use'], 'cell_id']
    d = df[df['cell_id'].isin(use_ids) & df['QD'].between(QD_MIN, QD_MAX)]
    d = d.sort_values(['cell_id', 'cycle']).copy()

    med = d.groupby('cell_id')['QD'].transform(
        lambda x: x.rolling(DEV_WINDOW, center=True, min_periods=1).median())
    spike = (d['QD'] - med).abs() > DEV_MAX
    return d[~spike].copy(), int(spike.sum())
