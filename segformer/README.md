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
