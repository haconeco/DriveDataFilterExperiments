# SegFormer 推論環境構築 TODO

以下は、RoadLib の SegFormer ベースモデルを用いて NuScenes 相当データから路面サインを検出・分類・ラベリングするために必要なタスク一覧です。

## 1. 依存調査
- [x] RoadLib リポジトリの `requirements.txt` / `environment.yml` / README から依存パッケージを抽出（PyTorch, MMCV/MMEngine, mmsegmentation, numpy, opencv-python, pycocotools など）。
  - リポジトリ: `https://github.com/GREAT-WHU/RoadLib`
  - 依存: `mmsegmentation`, `opencv-python`, `pcl`, `glfw3`, `ceres` (C++側), Python側は `mmseg` 依存。
- [x] CUDA 対応バージョンの確認（PyTorch + torchvision の互換表と、mmcv/mmengine のビルド要件を整理）。
- [x] NuScenes 入力を扱うためのユーティリティ有無確認（データローダや変換スクリプトの場所を特定）。
  - `nuscenes-devkit` を利用。

## 2. モデル・チェックポイント
- [x] 公開チェックポイントのダウンロード手順を確立（ダウンロード URL、検証用の SHA/MD5）。
- [x] RoadLib リポジトリ内の SegFormer 設定ファイル（config）を確認し、対応する checkpoint を配置するパスを決定。
- [x] 推論スクリプトが参照するパスの統一（例: `/workspace/weights/segformer.pth`）。

## 3. 入力・出力仕様
- [x] NuScenes 形式の 6 カメラ画像/フレームのパス構造を定義（例: `samples/CAM_FRONT/*.jpg` など）。
  - 入力パス: `samples/<CAMERA_NAME>/*.jpg` (例: `samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg`)
  - カメラ: `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`
- [x] 推論前処理（リサイズ、正規化、カメラ別の座標系合わせ）が RoadLib に存在するか確認し、無ければ補完ロジックを設計。
  - RoadLib `segformer_whu.py` 設定: `crop_size = (512, 1024)`, `mode='slide'`.
  - 推論は Perspective 画像に対して実行される。IPM は後処理（C++）で行われるが、Python スクリプトでは Perspective でのセグメンテーション結果を出力する。
- [x] 出力フォーマットを具体化：フレームごとに `[[{label, score, polygon/bbox, camera_id}, ...], ...]` のようなネストリスト形式に整理。
  - 出力: セグメンテーションマスク（PNG等）および抽出されたインスタンス情報（JSON）。
  - インスタンス抽出: `cv2.findContours` 等を用いてマスクからポリゴン/BBoxを生成する簡易ロジックを実装する。
- [x] ラベリング対象クラスの一覧を SegFormer 設定から抽出し、マッピング表を作成。
  - クラス定義 (`RoadLib/roadlib/roadlib.h` より):
    - 0: Background (EMPTY)
    - 1: SOLID (pixel value > 0, others)
    - 2: DASHED (pixel value 2)
    - 3: GUIDE (pixel value 3)
    - 4: ZEBRA (pixel value 4)
    - 5: STOP (pixel value 5)

## 4. Dockerfile 設計
- [x] ベースイメージ選定（例: `nvidia/cuda:<version>-cudnn-runtime-ubuntu20.04`）。
- [x] OS 依存パッケージ（git, wget, build-essential, python3-dev, ffmpeg 等）のインストール。
- [x] PyTorch + CUDA 対応バージョンのインストールコマンドを確定（pip/conda）。
- [x] mmcv/mmengine/mmsegmentation のインストール手順（適合する CUDA / Torch バージョンでのビルド or prebuilt wheel）。
- [x] RoadLib のクローン、必要なサブモジュール取得、環境変数設定。
- [x] チェックポイントとデータをマウント/配置するためのディレクトリ構成（`/workspace/data`, `/workspace/weights` 等）。

## 5. docker-compose 設計
- [x] GPU パススルー設定（`deploy.resources.reservations.devices` もしくは `runtime: nvidia`）。
- [x] ボリュームマウント（データ, 重みファイル, 出力先, ログ）。
- [x] 環境変数（PYTHONPATH, TORCH_HOME, MMCV_WITH_OPS ビルドフラグ等）の設定。
- [x] 推論エントリーポイント（例: `python tools/inference.py --config ... --checkpoint ... --input ... --output ...`）。

## 6. 推論スクリプト整備
- [x] NuScenes 6 カメラ画像をまとめて処理するラッパースクリプトを準備（ファイル列挙と一括前処理）。
- [x] SegFormer の出力をネストリスト形式に整形する関数を作成。
- [x] クラス名/スコア/位置情報（bbox or polygon）を含む JSON か Python リストで保存できるようにする。
- [x] サンプル入力と出力の例を `README.md` に追記予定。

## 7. 動作検証
- [x] 小規模な NuScenes 相当データセットで推論を実行し、フォーマット通りの出力を確認。
- [x] Dockerfile, docker-compose.yml のビルド・起動手順を検証し、再現性のあるログを残す。
- [x] 速度・メモリ使用量を確認し、必要なら Dockerfile での最適化（キャッシュ、不要ライブラリ削減）を検討。

## 8. ドキュメント整備
- [x] Dockerfile / docker-compose.yml の使い方を README に追記（ビルド・起動・入力配置・出力取得まで）。
- [x] 依存バージョン一覧と既知の制約（特定の CUDA バージョンのみ動作等）を明記。
- [x] トラブルシューティング（mmcv のビルド失敗、CUDA ランタイム不一致など）の FAQ を作成。
