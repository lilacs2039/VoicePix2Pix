# VoicePix2Pix
Pix2Pixを音声データに応用することで声質変換を目指すプロジェクト（Work in Progress）。





# Example

WIP




# Usage
## Google Colaboratoryを使用する場合

※データセットは初期設定では声優統計コーパスが使用されます。

1. 本リポジトリのUsage.ipynbをGoogle Colaboratoryから開く。↓のリンク先へ飛べばよい。
   https://colab.research.google.com/github/lilacs2039/VoicePix2Pix/blob/master/Usage.ipynb
2. GPUインスタンスを選択
3. すべてのセルを実行する。
   1. Ctrl+F9 or ランタイム>すべてのセルを実行
4. 実行結果はGoogleドライブの「マイドライブ＞Colab Notebooks＞VoicePix2Pix」フォルダ内に保存される。





## 任意のマシンで実行する場合

前提：chainerのmnist_cnnがGPUで学習できる状態

https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist.py



### インストール

以下のコマンドをプロジェクトを作成したいフォルダから実行すること

```bash
git clone --depth 1 https://github.com/lilacs2039/VoicePix2Pix
cd VoicePix2Pix
pip install -r requirements.txt

```



### データセットの準備

- datasetフォルダ内にtrainとtestフォルダを作成し、それぞれのフォルダ内にwavファイルを置いておく。あらかじめ10秒程度の長さの複数ファイルに分けておくとよい。

- データセット用のフォルダをもっているときは、以下コマンドをdatasetフォルダ内で実行してシンボリックリンクを作成してもよい



```
:windowsの場合
mklink /d train （wavファイルのあるディレクトリ）
mklink /d test （wavファイルのあるディレクトリ）
```

```bash
:linuxの場合
ln -s train （wavファイルのあるディレクトリ）
ln -s test （wavファイルのあるディレクトリ）
```



### 実行

```bash
# GPUありで学習
python train.py -g 0 -i ./dataset --out ./output --snapshot_interval 1000 --shared_mem 64000000

# CPUで学習する場合
#python train.py -g -1 -i ./dataset --out ./output --snapshot_interval 1000 --shared_mem 64000000


```





# 学習について

- 変換先の話者の音声データのみ必要
  - 変換元・変換先の話者のパラレルデータは不要
- 声質匿名化（教師音声から話者情報を削ぎ落す）
  - pyworldによって教師音声データから音響特徴量（基本周波数・スペクトル包絡・非周期性指標）を抽出。
  - 基本周波数を0で置き換えて、スペクトル包絡にガウシアンフィルタをかけたり一様分布ノイズを加えたりする。
  - 音響特徴量から音声を復元
- 声質匿名化した音声を入力データ、元の教師音声を対応する教師データとして学習を行う。
- 音声をPix2Pixモデルへ入れる前に音声をSTFTしてスペクトログラムと位相スペクトログラムへ変換し、２CHの画像として扱う。Pix2Pixモデルの入出力サイズ（行・列のサイズ）は実行時に動的に決定するため、音声の長さは任意（batchsize=1の場合）。
- 推論
  - 変換元の話者音声を声質匿名化してから、学習したモデルで推論を行う。



# パラメータチューニング方法

WIP



# 謝辞

本プログラムは「pfnet-research/chainer-pix2pix」を参考に作成しました。素晴らしいプログラムを公開くださり感謝いたします。

https://github.com/pfnet-research/chainer-pix2pix







