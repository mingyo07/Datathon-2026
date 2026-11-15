"""모델 입력 106개 = 원본 17 + 시계열 파생 50 + 이미지 밝기 통계 6 + 규칙 신호 33, 그리고 LightGBM 학습·예측과 후처리.

train.py와 test.py가 같은 함수를 호출한다. 모든 파생변수는 같은 실험의 "현재까지" 값만 쓰므로
행 순서, 다른 실험의 유무, 미래 행의 유무에 영향받지 않는다. 전처리 통계(평균, 표준편차 등)는 쓰지 않는다.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
WEIGHT_PATH = HERE / 'weights' / 'lgbm106.joblib'

CAT = ['bat_class', 'soc_status', 'tr_method']
RAW = ['relative_time_s', 'meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'meas_ch4', 'meas_co',
       'meas_co2', 'meas_hcl', 'meas_hf', 'meas_so2', 'atmp_t', 'thermal_t', 'bat_class',
       'bat_cc', 'soc_status', 'tr_method', 'indu_t']
RAW_NUM = [c for c in RAW if c not in CAT]
GAS6 = ['meas_co', 'meas_co2', 'meas_ch4', 'meas_hf', 'meas_hcl', 'meas_so2']
TEMPS = ['meas_t1', 'meas_t2', 'thermal_t']
EVENTS = ['vdrop', 'gas', 'hf', 'selfheat']

DERIVED = (
    ['v_drop_ratio', 'vdrop_flag', 't_since_vdrop', 'gas6_sum', 'gas_flag', 't_since_gas',
     'hf_flag', 't_since_hf', 't1_minus_indu', 'selfheat_flag', 't_since_selfheat']
    + [f'{c}_cummax' for c in TEMPS + GAS6 + ['meas_p']]
    + [f'{c}_rate30s' for c in ['meas_v', 'meas_p'] + TEMPS + GAS6]
    + [f'{c}_rate120s' for c in ['meas_v', 'meas_p'] + TEMPS]
    + ['t1_from_peak', 'thermal_from_peak', 't_since_t1_peak',
       't1_rate_cummax', 'thermal_rate_cummax', 'event_count']
    + ['t1_rise', 't1_rise_per_ah', 't1_minus_t2', 't1_minus_amb', 'thermal_minus_t1',
       'co_co2_ratio', 'hf_share']
)
IMG = ['img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99', 'img_hot_ratio']
# 규칙 신호 33개: 라벨 분석에서 찾은 단계 기준점(45℃, 전압 1/6 하락, 급가열, 정점)을 추적
PEAK_SIGNALS = ['meas_t1', 'meas_t2', 'thermal_t', 'tmax2', 'tmax3']
RULES = (
    ['v_drop_ratio_cummax', 'v_min_ratio_first', 's2_flag', 't_since_s2', 's4_flag', 't_since_s4',
     'tmax3_rate60s', 'tmax3_rate60s_cummax', 'run_flag', 't_since_run',
     't1_rise_after_s2', 't1_rate_since_s2']
    + [n for s in PEAK_SIGNALS
       for n in (f'{s}_cmax', f'{s}_drop_from_peak', f'{s}_peak_ratio', f't_since_{s}_peak')]
    + ['tmax2_minus_amb_cmax']
)
T_ON, V_ON, RUN_RATE = 45.0, 1 / 6, 1.0        # 2단계 진입 온도(℃), 4단계 전압 하락률, 급가열(℃/초)
# relative_time_s는 정렬·변화율·경과 시간 계산에만 쓰고 모델 입력에서는 뺀다
RAW_INPUT = [c for c in RAW if c != 'relative_time_s']
FEATURES = RAW_INPUT + DERIVED + IMG + RULES
assert len(RAW_INPUT) == 17 and len(DERIVED) == 50 and len(RULES) == 33 and len(FEATURES) == 106

SORT_KEYS = ['experiment_id', 'relative_time_s', 'sample_id']


def load_csv(path):
    dtype = {c: 'string' for c in ['sample_id', 'experiment_id'] + CAT}
    df = pd.read_csv(path, dtype=dtype)
    for c in RAW_NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    df[RAW_NUM] = df[RAW_NUM].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 이미지 밝기 통계 6개 (이미지 한 장씩 독립)
def image_paths(df, image_root):
    """CSV의 image_path를 실제 파일 경로로 바꾼다. --image_root가 이미지 폴더여도, 그 위 폴더여도 찾는다."""
    root, paths = Path(image_root), []
    for p, sid in zip(df['image_path'].astype(str), df['sample_id'].astype(str)):
        p = Path(p)
        for cand in (root / p, root / p.name, root / 'images' / p.name, root / f'{sid}.png', root.parent / p):
            if cand.is_file():
                paths.append(cand)
                break
        else:
            raise FileNotFoundError(f'이미지를 찾지 못했습니다: {p} (기준 폴더 {root})')
    return paths


def _one_image(path):
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return (a.mean(), a.std(), a.max(), np.percentile(a, 95), np.percentile(a, 99),
            (a >= 0.8).mean())


def image_stats(paths):
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(_one_image, paths))
    return np.asarray(rows, dtype=float)


def read_frame(csv, image_root):
    """센서 CSV를 읽고 이미지 밝기 통계 6개를 붙인다."""
    df = load_csv(csv)
    df[IMG] = image_stats(image_paths(df, image_root))
    return df


# ---------------------------------------------------------------- 시계열 파생 50개 + 규칙 신호 33개
def _cummax(x):
    return pd.Series(x).cummax().to_numpy()


def _pos(x):
    """0 이하인 값은 NaN으로(0이나 음수로 나누지 않게)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, np.nan)


def _first(x):
    """실험의 첫 관측값(결측이 아닌 첫 값)."""
    ok = np.flatnonzero(~np.isnan(x))
    return x[ok[0]] if len(ok) else np.nan


def _rate(t, x, window):
    """창 안에서 가장 오래된 관측과 현재 관측의 차이 / 실제 경과 초."""
    j = np.searchsorted(t, t - window, side='left')
    dt = t - t[j]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (x - x[j]) / dt
    r[~(dt > 0)] = np.nan
    return r


def _since_first(t, cond):
    """조건이 처음 참이 된 뒤로 플래그 1, 그 시각부터의 경과 초(그 전은 NaN)."""
    flag = np.zeros(len(t))
    since = np.full(len(t), np.nan)
    hit = np.flatnonzero(cond)
    if len(hit):
        i = hit[0]
        flag[i:] = 1.0
        since[i:] = t[i:] - t[i]
    return flag, since


def _group_features(g):
    """g: 한 실험, 시간순 정렬."""
    t = g['relative_time_s'].to_numpy(float)
    v = {c: g[c].to_numpy(float) for c in RAW_NUM}
    o = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        gas = g[GAS6].sum(axis=1).to_numpy(float)
        vmax = _cummax(v['meas_v'])
        o['v_drop_ratio'] = (vmax - v['meas_v']) / _pos(vmax)
        o['gas6_sum'] = gas
        o['t1_minus_indu'] = v['meas_t1'] - v['indu_t']

        conds = {
            'vdrop': o['v_drop_ratio'] > 0.10,
            'gas': gas - _first(gas) > 5,
            'hf': v['meas_hf'] - _first(v['meas_hf']) > 1,
            'selfheat': o['t1_minus_indu'] > 0,   # indu_t가 NaN(과충전)이면 항상 False
        }
        for e in EVENTS:
            o[f'{e}_flag'], o[f't_since_{e}'] = _since_first(t, conds[e])

        for c in TEMPS + GAS6 + ['meas_p']:
            o[f'{c}_cummax'] = _cummax(v[c])

        for c in ['meas_v', 'meas_p'] + TEMPS + GAS6:
            o[f'{c}_rate30s'] = _rate(t, v[c], 30)
        for c in ['meas_v', 'meas_p'] + TEMPS:
            o[f'{c}_rate120s'] = _rate(t, v[c], 120)

        o['t1_from_peak'] = o['meas_t1_cummax'] - v['meas_t1']
        o['thermal_from_peak'] = o['thermal_t_cummax'] - v['thermal_t']
        peak_t = pd.Series(np.where(v['meas_t1'] >= o['meas_t1_cummax'], t, np.nan)).ffill()
        o['t_since_t1_peak'] = t - peak_t.to_numpy()
        o['t1_rate_cummax'] = _cummax(o['meas_t1_rate30s'])
        o['thermal_rate_cummax'] = _cummax(o['thermal_t_rate30s'])
        o['event_count'] = sum(o[f'{e}_flag'] for e in EVENTS)

        o['t1_rise'] = v['meas_t1'] - _first(v['meas_t1'])
        o['t1_rise_per_ah'] = o['t1_rise'] / v['bat_cc']
        o['t1_minus_t2'] = v['meas_t1'] - v['meas_t2']
        o['t1_minus_amb'] = v['meas_t1'] - v['atmp_t']
        o['thermal_minus_t1'] = v['thermal_t'] - v['meas_t1']
        o['co_co2_ratio'] = v['meas_co'] / (v['meas_co2'] + 1)
        o['hf_share'] = v['meas_hf'] / (gas + 1)

        # ---- 규칙 신호 33개
        # 전압 규칙(3->4단계): 최고 전압 대비 1/6 이상 떨어진 적이 있는가
        o['v_drop_ratio_cummax'] = _cummax(o['v_drop_ratio'])
        o['v_min_ratio_first'] = pd.Series(v['meas_v']).cummin().to_numpy() / _first(v['meas_v'])
        o['s4_flag'], o['t_since_s4'] = _since_first(t, o['v_drop_ratio_cummax'] > V_ON)
        # 45℃ 규칙(1->2단계)과 그 이후의 진행
        o['s2_flag'], o['t_since_s2'] = _since_first(t, v['meas_t1'] > T_ON)
        o['t1_rise_after_s2'] = np.where(o['s2_flag'] > 0, o['meas_t1_cummax'] - T_ON, np.nan)
        o['t1_rate_since_s2'] = o['t1_rise_after_s2'] / _pos(o['t_since_s2'])
        # 급가열: 세 온도 중 최댓값이 60초 동안 초당 1℃ 넘게 오른 적이 있는가
        sig = dict(v)
        sig['tmax2'] = np.fmax(v['meas_t1'], v['meas_t2'])
        sig['tmax3'] = np.fmax(sig['tmax2'], v['thermal_t'])
        o['tmax3_rate60s'] = _rate(t, sig['tmax3'], 60)
        o['tmax3_rate60s_cummax'] = _cummax(o['tmax3_rate60s'])
        o['run_flag'], o['t_since_run'] = _since_first(t, o['tmax3_rate60s'] > RUN_RATE)
        # 정점 이후(5->6단계 후보)
        for s in PEAK_SIGNALS:
            cm = _cummax(sig[s])
            o[f'{s}_cmax'] = cm
            o[f'{s}_drop_from_peak'] = cm - sig[s]
            o[f'{s}_peak_ratio'] = sig[s] / _pos(cm)
            last_peak = pd.Series(np.where(sig[s] >= cm, t, np.nan)).ffill().to_numpy()
            o[f't_since_{s}_peak'] = t - last_peak
        o['tmax2_minus_amb_cmax'] = _cummax(sig['tmax2'] - v['atmp_t'])
    return pd.DataFrame(o, index=g.index)[DERIVED + RULES]


def add_features(df):
    """파생변수 50개와 규칙 신호 33개를 붙여 반환. 실험별로 시간순 정렬해 계산하고 입력 행 순서로 되돌린다."""
    df = df.reset_index(drop=True)
    s = df.sort_values(SORT_KEYS, kind='mergesort')
    parts = [_group_features(g) for _, g in s.groupby('experiment_id', sort=False)]
    feat = pd.concat(parts).reindex(df.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([df, feat], axis=1)


def thin_copies(df, k):
    """각 실험을 k개 중 1개만 남긴 복사본(5초 간격 -> 5k초 간격). 학습에서만 쓴다."""
    s = df.sort_values(SORT_KEYS, kind='mergesort')
    s = s[s.groupby('experiment_id').cumcount() % k == 0].copy()
    s['experiment_id'] = s['experiment_id'] + f'__s{k}'
    return s


# ---------------------------------------------------------------- 입력 행렬
def fit_cat_maps(df):
    """범주 -> 숫자 코드. 값이 숫자로 적혀 있으면(이 대회 데이터) 그 숫자를 그대로 쓴다. 학습 데이터로만 만든다."""
    maps = {}
    for c in CAT:
        vals = sorted(df[c].dropna().unique())
        try:
            maps[c] = {v: float(v) for v in vals}
        except ValueError:
            maps[c] = {v: float(i) for i, v in enumerate(vals)}
    return maps


def to_matrix(feat_df, cat_maps):
    X = feat_df[FEATURES].copy()
    for c in CAT:
        X[c] = X[c].map(cat_maps[c])          # 처음 보는 범주는 NaN(결측)으로
    return X.astype(float).to_numpy()


def build_inputs(df, cat_maps):
    """밝기 통계가 붙은 프레임 -> 모델 입력 (N, 106). 학습과 예측이 같은 함수를 쓴다."""
    return to_matrix(add_features(df), cat_maps)


# ---------------------------------------------------------------- LightGBM
def fit_model(X, y):
    import lightgbm as lgb
    # 학습 데이터에서 전부 결측이거나 값이 하나뿐인 열은 뺀다
    cols = np.array([j for j in range(X.shape[1]) if len(np.unique(X[:, j][~np.isnan(X[:, j])])) > 1])
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5,
                             subsample=0.8, subsample_freq=1, min_child_samples=30,
                             class_weight='balanced', random_state=42, n_jobs=8, verbose=-1)
    clf.fit(X[:, cols], y)
    return {'clf': clf, 'cols': cols}


def predict_proba(model, X):
    """단계 1~6 순서의 확률 (N, 6)."""
    clf = model['clf']
    p = np.zeros((len(X), 6))
    p[:, [int(c) - 1 for c in clf.classes_]] = clf.predict_proba(X[:, model['cols']])
    return p


# ---------------------------------------------------------------- 후처리
def monotone(df, pred):
    """실험별 시간순 누적 최댓값: 단계는 되돌아가지 않는다. 각 시점은 그 시점까지의 예측만 쓴다."""
    s = df[SORT_KEYS].reset_index(drop=True)
    s['pred'] = np.asarray(pred)
    order = s.sort_values(SORT_KEYS, kind='mergesort').index
    out = s.loc[order].groupby('experiment_id', sort=False)['pred'].cummax()
    return out.reindex(s.index).to_numpy().astype(int)


def predict_labels(bundle, df):
    """밝기 통계가 붙은 프레임 -> 단계 라벨(1~6)."""
    p = predict_proba(bundle['model'], build_inputs(df, bundle['maps']))
    return monotone(df, p.argmax(axis=1) + 1)
