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

## 現状
- まだ Dockerfile や docker-compose.yml は未作成。依存調査と設計方針のブレイクダウンは `TODO.md` にまとめています。

## 設計メモ（依存・リソース・ファイル構成）
### RoadLib 参照に基づくリスク洗い出し
- RoadLib 自体は C++ 実装で OpenCV/PCL/GLFW3/Ceres へリンクする前提のため、Python だけのコンテナでは再コンパイルや一部ツールの実行ができない（README の Installation 節より）。
- 付属の推論例 `scripts/inference_example.py` は GUI 表示 (`show=True`) と固定パス (`/home/zhouyuxuan/...`) を前提にしているため、ヘッドレス環境や我々のディレクトリ設計とはそのまま互換しない。
- 推論モデルの配布は外部 SharePoint で認証付きダウンロードとなっており、自動取得できない（手動配置を前提にする必要）。

上記を踏まえ、以下の設計修正と Dockerfile 修正方針を採用する。

- C++ 依存をコンテナに含め、必要に応じて RoadLib 本体のビルドやツール実行ができる状態を確保する。
- 推論スクリプトはヘッドレス運用（`show=False`）とパス上書きを前提にし、後続のラッパー実装でパス解決を行う。
- チェックポイントは `/workspace/weights/segformer_whu.pth` への手動配置を前提にする旨を明記し、自動ダウンロードは行わない。

### 依存関係の整理
- RoadLib が提供している SegFormer 設定（`scripts/segformer_whu.py`）は MMSegmentation のコンフィグをベースとしており、`mmseg.apis.MMSegInferencer` を呼び出しています。そのため PyTorch + MMCV/MMEngine/MMsegmentation の組み合わせを用意します（CUDA 11.8 + torch 2.1 系 + mmcv 2.1 系 + mmengine 0.10 系 + mmsegmentation 1.2 系を採用）。
- PyTorch まわり: `torch==2.1.2+cu118`, `torchvision==0.16.2+cu118`, `torchaudio==2.1.2+cu118` を公式の CUDA 11.8 wheels から取得。
- OpenMMLab 依存: `mmengine==0.10.4`, `mmcv==2.1.0`, `mmsegmentation==1.2.2`（いずれも CUDA 11.8 / torch 2.1 と互換）を `openmim` 経由でインストール。
- 追加 Python 依存: `numpy`, `opencv-python`, `pycocotools`, `pillow`, `scipy`, `tqdm`, `rich`, `matplotlib`, `nuscenes-devkit`（NuScenes 入力ユーティリティ確保）。
- OS パッケージ: `build-essential`, `python3-dev`, `git`, `wget`, `curl`, `ffmpeg`, `libgl1`, `libglib2.0-0`, `ca-certificates` をベースでインストール。RoadLib 本体のビルドも想定し、`libopencv-dev`, `libpcl-dev`, `libglfw3-dev`, `libceres-dev`, `libeigen3-dev`, `pkg-config` を追加。

### 必要リソースと取得方針
- SegFormer 用コンフィグ: RoadLib 付属の `scripts/segformer_whu.py` をビルド時に `curl` で取得し、MMSegmentation の `configs/segformer/` に配置。
- チェックポイント: RoadLib README で案内されている公開モデル（SharePoint 配布）を `/workspace/weights/segformer_whu.pth` に配置する想定。配布元が認証付きのため、ビルド時ダウンロードは行わず、利用者が手元でダウンロードしてマウントする運用を前提とする。
- 補助スクリプト: `scripts/inference_example.py` を参考に、推論実行時は `MMSegInferencer` にコンフィグ/ウェイト/入力画像のパスを渡す。

### 入出力ディレクトリ設計
- `/workspace/mmsegmentation`: MMSegmentation ソースを配置し、`pip install -e` でインストール。`configs/segformer/segformer_whu.py` をここに格納。
- `/workspace/RoadLib`: RoadLib リポジトリ本体。SegFormer 以外の補助スクリプトやデータ定義を参照するためにソースごと取得。
- `/workspace/data/nuscenes`: 入力データ格納ルート（例: `samples/CAM_FRONT/*.jpg`, `samples/CAM_FRONT_LEFT/*.jpg` など 6 カメラをサブディレクトリで保持）。
- `/workspace/weights`: 学習済みチェックポイント格納ルート。デフォルトは `segformer_whu.pth` を置く。
- `/workspace/output`: 推論結果（可視化・JSON など）の出力先。

### コンテナイメージ設計の概要
- ベース: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` を採用し、CUDA 11.8 + PyTorch 2.1 系を基盤にする。
- Python 環境: システム Python3 を利用し、`pip` をアップグレードした上で PyTorch → OpenMMLab 依存を順にインストール。`openmim` を使って mmcv/mmengine/mmsegmentation のバージョン整合性を確保。
- ソース配置: ビルド時に MMSegmentation（v1.2.2）と RoadLib をクローンし、`PYTHONPATH=/workspace/mmsegmentation:/workspace/RoadLib` を設定。SegFormer コンフィグは MMSegmentation 側の `configs/segformer/` にダウンロードしておく。必要に応じて RoadLib C++ 部分をビルドできるよう、依存ライブラリをインストール済みにする。
- ビルド生成物: `/workspace/data`, `/workspace/weights`, `/workspace/output` を作成し、チェックポイントや入力データはホスト側からボリュームマウントする運用。推論エントリーポイント例として `python tools/inference.py --config configs/segformer/segformer_whu.py --checkpoint /workspace/weights/segformer_whu.pth --input /workspace/data/nuscenes/samples --output /workspace/output` を想定。
