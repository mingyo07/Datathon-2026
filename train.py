"""학습: Train + Validation 97개 실험으로 LightGBM을 학습해 weights/lgbm106.joblib 로 저장한다. Test는 읽지 않는다.

    python train.py --data_dir /home/ubuntu/data/battery_data_light             # 제출용 가중치 생성
    python train.py --data_dir /home/ubuntu/data/battery_data_light --validate  # Train만 학습 -> Validation 점수(가중치 저장 안 함)
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, default=Path('/home/ubuntu/data/battery_data_light'),
                    help='train/, validation/ 이 들어 있는 폴더')
    ap.add_argument('--validate', action='store_true', help='Train만으로 학습해 Validation의 5·10·15초 Macro F1을 출력')
    args = ap.parse_args()
    t0 = time.time()

    frames = {s: M.read_frame(args.data_dir / s / f'{s}.csv', args.data_dir / s) for s in ('train', 'validation')}
    if set(frames['train'].experiment_id) & set(frames['validation'].experiment_id):
        raise ValueError('Train과 Validation에 같은 실험이 있습니다')
    full = frames['train'] if args.validate else pd.concat([frames['train'], frames['validation']], ignore_index=True)
    maps = M.fit_cat_maps(full)

    # 학습 행: 원본(5초 간격) + 각 실험을 2행·3행마다 1행만 남긴 복사본(10초·15초 간격).
    # 복사본은 별개 실험으로 취급하고 솎은 뒤에 파생변수를 다시 계산한다(Test에 10초·15초 간격 실험이 있다).
    xs, ys = [], []
    for view in (full, M.thin_copies(full, 2), M.thin_copies(full, 3)):
        view = view.reset_index(drop=True)
        xs.append(M.build_inputs(view, maps))
        ys.append(view.label.to_numpy(int))
    X, y = np.vstack(xs), np.concatenate(ys)
    model = M.fit_model(X, y)
    print(f'학습 행 {len(y):,} (원본 {len(ys[0]):,} + 복제본 {len(y) - len(ys[0]):,}), 변수 {len(model["cols"])}개', flush=True)

    if args.validate:
        from sklearn.metrics import f1_score
        val, scores = frames['validation'], []
        print('Validation (Train만 학습, 누적 최댓값 보정 후)   단계별 F1 1~6 | Macro F1')
        for k in (1, 2, 3):
            view = (val if k == 1 else M.thin_copies(val, k)).reset_index(drop=True)
            pred = M.predict_labels({'model': model, 'maps': maps}, view)
            per = f1_score(view.label.to_numpy(int), pred, labels=[1, 2, 3, 4, 5, 6], average=None, zero_division=0)
            scores.append(per.mean())
            print(f'{5 * k:>2}초  ' + '  '.join(f'{s:.4f}' for s in per) + f'  | {per.mean():.4f}', flush=True)
        print(f'세 간격 평균 Macro F1 {np.mean(scores):.4f}')
        return

    M.WEIGHT_PATH.parent.mkdir(exist_ok=True)
    tmp = M.WEIGHT_PATH.with_suffix('.tmp')
    joblib.dump({'model': model, 'maps': maps, 'features': M.FEATURES}, tmp)
    tmp.replace(M.WEIGHT_PATH)
    print(f'저장: {M.WEIGHT_PATH}  ({M.WEIGHT_PATH.stat().st_size / 1e6:.1f}MB, {time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
