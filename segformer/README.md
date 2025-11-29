# SegFormer ベースの路面サイン検出/分類/ラベリング概要

このディレクトリでは、GREAT-WHU が公開している RoadLib の SegFormer ベースモデルを用いて、NuScenes 相当の入力データ（6 カメラ画像/フレーム）から路面上のサインを検出・分類・ラベリングするための環境構築を行います。

## 目的
- NuScenes 形式のシーンを入力とし、フレーム単位で 6 枚の画像を処理。
- 検出・分類結果を、フレームごとに「写っている対象」を列挙したネストしたリスト形式で出力。
- 既存の公開リポジトリ（[RoadLib](https://github.com/GREAT-WHU/RoadLib)）と提供済みの学習済みチェックポイントを活用して、推論用のコンテナ環境を構築。

## コンテナ構成のゴール
- RoadLib/SegFormer 推論が可能な Docker イメージと docker-compose 定義を用意する。
- 依存ライブラリ（PyTorch、MMCV/MMEngine、NuScenes データローダー周辺など）を明示し、ビルド手順を自動化する。
- 推論スクリプトを通じて、入力（6 カメラ画像/フレーム）→ 出力（ラベル付きのネストリスト）の最小パイプラインを再現できるようにする。

## 参考リソース
- RoadLib リポジトリ: https://github.com/GREAT-WHU/RoadLib
- 公開チェックポイント: https://whueducn-my.sharepoint.com/personal/2015301610143_whu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F2015301610143%5Fwhu%5Fedu%5Fcn%2FDocuments%2FRoadLib%2Finference&ga=1

## 使い方

### 1. 環境構築 (Build)
Docker イメージをビルドします。
```bash
cd segformer
docker-compose build
```

### 2. 動作検証 (Test)
Unit Test を実行して、ロジック（インスタンス抽出など）が正しく動作することを確認します。
```bash
docker-compose run --rm segformer python3 -m unittest discover tests
```

### 3. 推論実行 (Inference)
NuScenes データセットに対して推論を実行します。
```bash
docker-compose run --rm segformer python3 tools/inference.py \
  --config mmsegmentation/configs/segformer/segformer_whu.py \
  --checkpoint weights/segformer.pth \
  --dataroot data/nuscenes \
  --version v1.0-mini \
  --output_dir output/run_01 \
  --visualize
```

### 出力
- **JSON 結果**: `output/run_01/results.json`
- **マスク画像**: `output/run_01/masks/*.png`
- **可視化画像**: `output/run_01/vis/*.jpg` (Overlay)

## ディレクトリ構成
- `Dockerfile`: 環境定義 (CUDA 11.8, PyTorch 2.1, MMSegmentation 1.2.2)
- `docker-compose.yml`: GPU設定、ボリュームマウント定義
- `tools/inference.py`: 推論スクリプト
- `tests/`: Unit Test
- `weights/`: モデルチェックポイント配置場所 (手動配置)
- `output/`: 出力先
