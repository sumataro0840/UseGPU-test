# pytorch-test Boot Method

PyTorch + CUDA + 画像認識 + 動画の物体検出を試すための操作メモ。

## 前提

- Windows + WSL2
- VS Code Remote SSH または WSL 上で作業
- Python の仮想環境 `.venv` を使う
- 学習済みモデル、動画、抽出画像、ラベルデータは Git に入れない

## フォルダ構成

```text
pytorch-test/
├── data/
│   ├── images/                         # 画像認識用の入力画像
│   ├── videos/                         # 入力動画
│   ├── results/
│   │   └── videos/                     # 検出済み動画
│   └── datasets/
│       └── shuttlecock/
│           ├── images/
│           │   ├── unlabeled/          # 動画から切り出した未ラベル画像
│           │   ├── train/              # 学習用画像
│           │   └── val/                # 検証用画像
│           ├── labels/
│           │   ├── train/              # 学習用YOLOラベル
│           │   └── val/                # 検証用YOLOラベル
│           └── data.yaml
├── models/                             # 学習済みモデル
├── src/
├── main.py
├── requirements.txt
└── README.md
```

## 初回セットアップ

システムPythonへ直接 `pip install` すると `externally-managed-environment` が出ることがあるので、このプロジェクト専用の仮想環境を使う。

```bash
cd ~/pytorch-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Label Studio も同じ仮想環境に入れる。

```bash
pip install label-studio
```

## 2回目以降の起動

```bash
cd ~/pytorch-test
source .venv/bin/activate
```

画像認識:

```bash
python3 main.py data/images/sample.jpg
```

動画の物体検出:

```bash
python3 main.py data/videos/sample.mp4
```

結果は `data/results/videos/` に保存される。

## Label Studio の起動

リモートPC上で Label Studio を起動する。

```bash
cd ~/pytorch-test
source .venv/bin/activate
label-studio start --host 127.0.0.1 --port 8080
```

SSHポートフォワーディングで接続している場合は、手元PCのブラウザで開く。

```text
http://localhost:8080
```

SSHし直す場合の例:

```bash
ssh -L 8080:localhost:8080 suma@100.107.212.43
```

## Label Studio のラベル設定

プロジェクト作成後、Labeling Interface の Code に以下を入れて Save する。

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="shuttlecock"/>
  </RectangleLabels>
</View>
```

ラベル付けでは、シャトルだけをできるだけ正確に四角で囲む。

- シャトルが見える画像だけ使う
- シャトル以外の天井、看板、照明、人物、コートを囲まない
- ブレていてもシャトルと分かるなら囲む
- 見切れている場合は見えている部分だけ囲む
- 分からない画像は無理にラベル付けしない

## シャトル検出の運用フロー

### 1. 動画を置く

入力動画をここに置く。

```text
data/videos/badminton.mp4
```

### 2. フレーム抽出

Label Studio でラベル付けするための画像を動画から切り出す。

```bash
cd ~/pytorch-test
source .venv/bin/activate
python3 main.py --mode extract-frames --frame-step 15 data/videos/badminton.mp4
```

出力先:

```text
data/datasets/shuttlecock/images/unlabeled/badminton/
```

`--frame-step 15` は15フレームに1枚だけ保存する設定。細かく取りたい場合は `5` や `10` にする。

### 3. Label Studio に画像を入れる

Label Studio の Data Import で、以下の画像を入れる。

```text
data/datasets/shuttlecock/images/unlabeled/badminton/
```

ブラウザが手元PCで、画像がSSH先にある場合は、VS Code の Explorer から対象フォルダを右クリックして Download するか、手元PCに `scp` でコピーしてからアップロードする。

例:

```bash
scp -r suma@100.107.212.43:~/pytorch-test/data/datasets/shuttlecock/images/unlabeled/badminton ./badminton_frames
```

### 4. Label Studio でラベル付け

各画像で `shuttlecock` を選び、シャトルだけを四角で囲む。

最低でもまずは100枚程度を目標にする。現在のように30枚前後だと動画ではかなり外れやすい。

### 5. YOLO形式でExport

Label Studio の Export から YOLO 形式で出力する。

ZIPの中に `labels/` があり、画像が入っていない場合でも問題ない。元画像は `images/unlabeled/` にあるものを使う。

### 6. 学習データを配置

最終的にこの形にする。

```text
data/datasets/shuttlecock/
├── images/
│   ├── train/
│   │   ├── sample_000000.jpg
│   │   └── ...
│   └── val/
│       ├── sample_000240.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── sample_000000.txt
│   │   └── ...
│   └── val/
│       ├── sample_000240.txt
│       └── ...
└── data.yaml
```

重要:

- 画像とラベルは同じ名前にする
- `images/train/sample_000000.jpg` に対応するラベルは `labels/train/sample_000000.txt`
- `train` に8割、`val` に2割くらいで分ける
- Label Studio が `ee885f05-sample_000000.txt` のような名前を出した場合は、元画像と同じ `sample_000000.txt` に直す

### 7. 学習開始

軽く動作確認する場合:

```bash
python3 main.py --mode train-shuttle --epochs 5
```

本番寄りに学習する場合:

```bash
python3 main.py --mode train-shuttle --epochs 50
```

成功すると以下が作られる。

```text
models/shuttlecock.pt
```

### 8. 動画にシャトル検出をかける

まず軽く試す。

```bash
python3 main.py --model models/shuttlecock.pt --confidence 0.05 --image-size 960 data/videos/badminton.mp4
```

シャトルが小さくて見つからない場合:

```bash
python3 main.py --model models/shuttlecock.pt --confidence 0.03 --image-size 1280 data/videos/badminton.mp4
```

ただし `--image-size 1280` はかなり遅い。長い動画では数分以上かかる。

出力先:

```text
data/results/videos/badminton_detected.mp4
```

## 結果動画を見る

VS Code の Explorer で `data/results/videos/` を開き、動画を Download する。

または手元PCのターミナルでコピーする。

```bash
scp suma@100.107.212.43:~/pytorch-test/data/results/videos/badminton_detected.mp4 .
```

## 精度が悪いときの確認

シャトルが囲われない場合は、だいたい以下が原因。

- ラベル数が少ない
- シャトル以外を囲っている
- 画像内のシャトルが小さすぎる
- 似た背景や照明をシャトルとして覚えている
- `val` にシャトル注釈が少なすぎる

まずはラベル品質を確認する。赤枠付きの確認画像を作る場合は、必要に応じて `data/results/debug/` に出す。

## Git に入れないもの

以下は `.gitignore` で除外する。

- 入力動画
- 出力動画
- 学習済みモデル `*.pt`
- Label Studio の export ZIP
- YOLO の `runs/`
- 学習用の実画像とラベル

Git に入れるのは、コード、設定ファイル、空フォルダ維持用の `.gitkeep`、`data.yaml` などだけにする。
