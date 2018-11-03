import scipy
from chainer.dataset import dataset_mixin
import numpy as np
import pyworld as pw
import glob
import librosa
import os

import parameters


class Vp2pDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir):
        print("search dataset paths")
        print("    from: %s"%dataDir)
        self.dataPaths = glob.glob(os.path.abspath(dataDir)+"/*.wav")
        print("load dataset paths done")

    def __len__(self):
        return len(self.dataPaths)

    @staticmethod
    def _ScaleArray( ndArray, min=0, max=1):
        """
        ndArrayの全データがmin~maxの範囲内になるように値をスケールする
        :param ndArray:
        :param min:
        :param max:
        :return: スケール後の配列、スケール倍率、オフセット
        """
        offset = - np.min(ndArray)
        rectified = ndArray + offset
        scale_factor = (max-min) / (np.max(rectified) - np.min(rectified))
        scaled = rectified * scale_factor + min
        return scaled, scale_factor,offset

    @staticmethod
    def _formatSound(fs,data, f0Value = 0, sp_strechRatio = np.random.uniform(0.6, 2, size=1),gaussian_s = 3):
        """
        WAV音声データから話者情報を取り除いたスペクトログラム・位相（2ch画像データ）を作成
        :param path:
        :param f0Value:
        :param sp_strechRatio:
        :return:
        """
        data = data.astype(np.float)
        _f0, t = pw.dio(data, fs)  # 基本周波数の抽出
        f0 = pw.stonemask(data, _f0, t, fs)  # 基本周波数の修正
        sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
        ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出
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

    @staticmethod
    def convert_to_spectrogram(waveNDArray):
        # スペクトル・位相マップ　作成
        D = librosa.stft(waveNDArray,n_fft=parameters.n_fft)  #D:np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        # スペクトログラムの行列サイズをEncoderに特徴的な値の整数倍にする
        # Encが8レイヤ、各レイヤで行列サイズ1/2になるので、入力スペクトログラムの行・列のサイズは256の倍数とする
        D = D[0:D.shape[0]-1,:]  #最後の行を削除
        w_div,w_rem = divmod(D.shape[1],parameters.Encocer_Feature_Constant)
        D = np.pad(D,[(0,0),(0,parameters.Encocer_Feature_Constant*(w_div+1) - D.shape[1])],
                   'constant',constant_values = np.min(np.abs(D)))
        Dabs = np.log10(np.abs(D))
        Dphase = np.angle(D)
        return Dabs,Dphase

    def get_example(self,i):
        path = self.dataPaths[i]
        label ,fs= librosa.load(path,sr=parameters.sample_ratio)
        input = self._formatSound(fs,label)
        label_abs, label_phase = self.convert_to_spectrogram(label)
        input_abs,input_phase = self.convert_to_spectrogram(input)
        label_abs,_scale_factor,_offset= self._ScaleArray(label_abs)
        input_abs,_scale_factor,_offset= self._ScaleArray(input_abs)
        return np.array([label_abs,label_phase]),np.array([input_abs,input_phase])

    def get_pathName(self,i):
        return self.dataPaths[i]
