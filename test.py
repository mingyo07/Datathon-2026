"""추론: 저장된 가중치로 예측만 한다(재학습 없음, Test로 맞추는 전처리 없음).

    python test.py --input <CSV> --image_root <이미지 폴더> --output <결과 CSV>
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--image_root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--weights', type=Path, default=M.WEIGHT_PATH)
    args = ap.parse_args()

    bundle = joblib.load(args.weights)
    if bundle['features'] != M.FEATURES:
        raise ValueError('가중치를 만든 코드와 model.py의 입력 열 목록이 다릅니다')
    df = M.read_frame(args.input, args.image_root)          # 센서 CSV + 이미지 밝기 통계 6개
    labels = M.predict_labels(bundle, df)                   # 입력 106개 -> LightGBM -> 실험별 누적 최댓값
    out = pd.DataFrame({'sample_id': df['sample_id'].to_numpy(), 'label': labels.astype(int)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'{len(out):,}행 저장: {args.output}   단계 분포: '
          + ', '.join(f'{k}:{v}' for k, v in out['label'].value_counts().sort_index().items()), flush=True)


if __name__ == '__main__':
    main()
