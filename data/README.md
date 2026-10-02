# data

데이터 파일은 용량이 커서(약 8GB) 저장소에 올리지 않는다. 아래 파일을 받아 이 폴더에 둔다.

## 원본 파일

MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019). 과제에서 지정한 배포처: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle

| 역할 | 파일 | 크기 | 전체 셀 |
|---|---|---:|---:|
| 학습 (Batch 1) | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3.0GB | 46 |
| 평가 (Batch 2) | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 2.0GB | 47 |
| 추가 평가 (Batch 3) | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3.2GB | 46 |

파일은 MATLAB v7.3(HDF5) 형식이며 `mat73`으로 읽는다.

## 가공 캐시

원본을 매번 읽으면 배치당 약 20초가 걸리므로, 필요한 값만 뽑아 `data/processed/`에 저장해 두고 쓴다.

| 파일 | 내용 |
|---|---|
| `summary.parquet` | 사이클별 요약값 116,722행 (방전 용량, 충전 용량, 내부 저항, 온도, 충전 시간) |
| `qdlin.npz` | 셀별 초기 101개 사이클의 방전 곡선 `Qdlin` (139셀 × 101 × 1,000)과 전압 축 `Vdlin` |

저장소 최상위에서 아래를 한 번 실행하면 만들어진다(약 1분).

```bash
python -c "from src.preprocess import build_cache; build_cache('data', 'data/processed')"
```

## 폴더 구조

```
data/
├── README.md
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
├── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
└── processed/
    ├── summary.parquet
    └── qdlin.npz
```
