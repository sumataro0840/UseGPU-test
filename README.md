# UseGPU-test

PyTorch + CUDA + GPU を使って、数値計算・画像認識・動画の物体検出を試すための練習用プロジェクトです。

## 背景

WSL2 + Docker + CUDA + PyTorch の環境構築が完了し、GPU が正常に認識されることを確認しました。
実際に PyTorch から `cuda:0` 上でテンソル演算が実行され、行列サイズを大きくした場合には CPU より GPU の方が大幅に高速であることも確認済みです。

今後は単なる環境確認に留まらず、GPU を使ってさまざまな処理や実験を行える状態を整えていきます。

## 方針

現状では、特定の用途を一つに決め打ちするというよりも、まずは GPU を使って何ができるのかを広く試します。

関心がある方向:

- PyTorch を用いた数値計算や機械学習
- 大規模な行列演算や並列計算の体験
- ノイズ除去や信号処理のような工学寄りのテーマ
- CUDA を用いた GPU 計算の基礎理解
- 画像認識や動画の物体検出
- 将来的なローカル LLM 推論や推論高速化の検証

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 画像認識

まず画像を `data/images/sample.jpg` に置いてください。

```bash
python3 main.py
```

別の画像を指定する場合:

```bash
python3 main.py data/images/your_image.jpg
```

## 動画の物体検出

動画を `data/videos/` に置いて実行します。

```bash
python3 main.py data/videos/sample.mp4
```

検出済み動画は `data/results/videos/` に保存されます。

## シャトル検出

バドミントンのシャトルを認識したい場合は、既成の `yolov8n.pt` ではなくシャトル用モデルを学習します。

まず動画からラベル付け用のフレームを切り出します。

```bash
python3 main.py --mode extract-frames --frame-step 15 data/videos/badminton.mp4
```

切り出した画像の中からシャトルが見えるものを選び、YOLO形式で `shuttlecock` としてラベル付けします。学習用データはこの形に置きます。

```text
data/datasets/shuttlecock/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

ラベル付け後に学習します。

```bash
python3 main.py --mode train-shuttle --epochs 50
```

学習が終わると `models/shuttlecock.pt` が作られます。その後、動画に対してシャトル検出を実行します。

```bash
python3 main.py --model models/shuttlecock.pt --confidence 0.2 data/videos/badminton.mp4
```

実行入口は `main.py` だけにして、画像認識の処理は `src/image_recognition/`、動画の物体検出は `src/object_detection/` のクラスに分けています。
