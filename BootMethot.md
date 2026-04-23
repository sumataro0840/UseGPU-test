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