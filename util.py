#!/usr/bin/env python
"""
モジュール間をまたいで共有される変数と共通の関数
"""
import librosa
import numpy as np
import pyworld as pw
import scipy



"""
wavデータのサンプリングレート
"""
sample_ratio = 22050

"""
Short Time FFT の窓サイズ

librosa規定値　2048
stftの戻り値の形状　(1 + n_fft/2, t)
"""
n_fft = 2048


"""
Encoderで縦・横方向に畳み込むサイズ倍率
Encが8レイヤ、各レイヤで行列サイズ1/2になるので 256

入力スペクトログラムの行・列のサイズはこの倍数とすること
"""
Encocer_Feature_Constant=2**8

"""
ラベル画像の保存をするか
"""
enable_output_labelWav = True

"""
scaleArray()のscaleFactorを表示するか
"""
print_scaleFactor=False

"""
rescaleArrayのデフォルトのscale_factor
データセット音声に合わせて変更のこと
"""
scale_factor=0.1

"""
rescaleArrayのデフォルトのoffset
データセット音声に合わせて変更のこと
"""
offset=0.8

"""
U-NETのチャネル自由度　Degree of freedom
"""
DOF = [16,32,64,128,128,128,128,128]  #[32,64,128,256,256,256,256,256]  #[64,128,256,512,512,512,512,512]

"""
各音声データの長さ
ミニバッチ学習するならデータ長さ統一する必要あり
None:データ長さ統一しない
"""
audio_dataset_second = None


def convert_to_wave(Dabs, Dphase):
    D_hat = 10 ** Dabs * np.exp(1j*Dphase)    #xp.exp(1j*Dphase)
    y_hat = librosa.istft(D_hat)
    return y_hat

def convert_to_spectrogram(waveNDArray):
    # スペクトル・位相マップ　作成
    D = librosa.stft(waveNDArray, n_fft=n_fft)  #D:np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
    # スペクトログラムの行列サイズをEncoderに特徴的な値の整数倍にする
    # Encが8レイヤ、各レイヤで行列サイズ1/2になるので、入力スペクトログラムの行・列のサイズは256の倍数とする
    D = D[0:D.shape[0]-1,:]  #最後の行を削除
    w_div,w_rem = divmod(D.shape[1], Encocer_Feature_Constant)
    D = np.pad(D, [(0,0), (0, Encocer_Feature_Constant * (w_div + 1) - D.shape[1])],
               'constant', constant_values = np.min(np.abs(D)))
    Dabs = np.log10(np.abs(D) + 10**-10)
    Dphase = np.angle(D)
    return Dabs,Dphase


def generate_inputWave(fs, waveNDArray, f0Value = 0, sp_strechRatio = np.random.uniform(0.6, 2, size=1), gaussian_s = 3):
    """
    WAV音声データから話者情報を取り除いたWAV音声データを作成
    label音声からinput音声作成用
    :param path:
    :param f0Value:
    :param sp_strechRatio:
    :return:
    """
    waveNDArray = waveNDArray.astype(np.float)
    _f0, t = pw.dio(waveNDArray, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(waveNDArray, _f0, t, fs)  # 基本周波数の修正
    sp = pw.cheaptrick(waveNDArray, f0, t, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(waveNDArray, f0, t, fs)  # 非周期性指標の抽出
    f0_fixed0 = np.ones(f0.shape) * f0Value
    f0_median = np.median(f0)
    sp_median = np.median(sp)
    ap_median = np.median(ap)
    # SPを高周波方向に伸縮
    sp2 = np.ones_like(sp)*np.min(sp)
    for f in range(sp2.shape[1]):
        if(int(f / sp_strechRatio) >= sp.shape[1]): break
        sp2[:, f] = sp[:, int(f / sp_strechRatio)]
    # SP/APに正規分布ノイズ
    sp_noised = sp2 + np.random.normal(sp_median,sp_median/10,sp2.shape)
    ap_noised = ap + np.random.normal(ap_median,ap_median/10,ap.shape)
    #ガウシアンフィルタ
    sp_gaussian = scipy.ndimage.filters.gaussian_filter(sp_noised,gaussian_s)
    ap_gaussian = scipy.ndimage.filters.gaussian_filter(ap_noised,gaussian_s)
    # 音声復元
    synthesized = pw.synthesize(f0_fixed0, sp, ap, fs)
    return synthesized


def scaleArray(ndArray, min=0, max=1):
    """
    ndArrayの全データがmin~maxの範囲内になるように値をスケールする
    :param ndArray:
    :param min:
    :param max:
    :return: スケール後の配列、スケール倍率、オフセット
    """
    scale_factor = (max-min) / (np.max(ndArray) - np.min(ndArray))
    scaled = ndArray * scale_factor
    offset = - np.min(scaled) + min
    ret = scaled + offset
    if print_scaleFactor:
        print('scale:{}, offset:{}'.format(scale_factor,offset))
    return ret, scale_factor,offset

def rescaleArray(ndArray, scale_factor=scale_factor,offset=offset):
    rescaled = (ndArray - offset)/ scale_factor
    return rescaled

def clip_audio_length(audio_ndarray, sr):
    """
    audio_ndarray の長さをdata_lengthに変更する。
    :param audio_ndarray:
    :param sr:
    :return:
    """
    if audio_ndarray.shape[0] > audio_dataset_second:
        ret = audio_ndarray[:audio_dataset_second * sr]
    else:
        ret = np.pad(audio_ndarray, [(0, audio_dataset_second * sr - audio_ndarray.shape[0])], 'constant', constant_values=0)
    return ret
