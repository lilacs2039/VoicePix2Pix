from chainer.dataset import dataset_mixin
import numpy as np
import glob
import librosa
import os

import util
from util import convert_to_spectrogram, generate_inputWave, scaleArray


class Vp2pDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir):
        print("search dataset paths")
        print("    from: %s"%dataDir)
        self.dataPaths = glob.glob(os.path.abspath(dataDir)+"/*.wav")
        print("load dataset paths done")

    def __len__(self):
        return len(self.dataPaths)

    def get_example(self,i):
        path = self.dataPaths[i]
        label ,fs= librosa.load(path, sr=util.sample_ratio)
        input = generate_inputWave(fs, label)
        label_abs, label_phase = convert_to_spectrogram(label)
        input_abs,input_phase = convert_to_spectrogram(input)
        label_abs,_scale_factor,_offset= scaleArray(label_abs)
        input_abs,_scale_factor,_offset= scaleArray(input_abs)
        return np.array([label_abs,label_phase]),np.array([input_abs,input_phase])

    def get_pathName(self,i):
        return self.dataPaths[i]


