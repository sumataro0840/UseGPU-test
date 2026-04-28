# pytorch-test

PyTorch + Docker + WSL2 + CUDA の練習用フォルダ。

## 前提

- Windows + WSL2
- Docker Desktop
- NVIDIA GPU が Docker から使える
- `docker run --gpus all ...` が通る
- PyTorch から `torch.cuda.is_available()` が `True`

---

## フォルダ構成

```bash
pytorch-test/
├── data/
├── models/
├── notebooks/
├── src/
├── main.py
└── README.md
```
## 一発起動用コマンド

毎回コンテナに入らずにスクリプトを直接実行できる。

```bash
cd ~/pytorch-test && docker run --rm --gpus all \
-v $PWD:/workspace \
-w /workspace \
pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime \
python main.py
```

## ローカルPythonで実行する場合

システムPythonへ直接 `pip install` すると `externally-managed-environment` が出ることがあるので、このプロジェクト専用の仮想環境を使う。

初回だけ:

```bash
cd ~/pytorch-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2回目以降:

```bash
cd ~/pytorch-test
source .venv/bin/activate
python3 main.py
```

動画の物体検出:

```bash
cd ~/pytorch-test
source .venv/bin/activate
python3 main.py data/videos/sample.mp4
```

結果は `data/results/videos/` に保存される。

## シャトル検出の進め方

既成のYOLOモデルにはシャトル専用クラスがないので、手持ち動画からシャトル用モデルを作る。

1. フレーム抽出

```bash
cd ~/pytorch-test
source .venv/bin/activate
python3 main.py --mode extract-frames --frame-step 15 data/videos/badminton.mp4
```

2. 抽出した画像をラベル付け

`data/datasets/shuttlecock/images/unlabeled/` に出た画像から、シャトルが見えるものを選んで YOLO 形式でラベル付けする。クラス名は `shuttlecock`。

置き場所:

```bash
data/datasets/shuttlecock/images/train/
data/datasets/shuttlecock/images/val/
data/datasets/shuttlecock/labels/train/
data/datasets/shuttlecock/labels/val/
```

3. 学習

```bash
python3 main.py --mode train-shuttle --epochs 50
```

4. シャトル検出

```bash
python3 main.py --model models/shuttlecock.pt --confidence 0.2 data/videos/badminton.mp4
```
