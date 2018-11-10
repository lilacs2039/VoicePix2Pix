import glob
import os
import sys

import librosa
import numpy as np
import unittest
import data_loader
import matplotlib.pyplot as plt

import util
from util import scaleArray


class TestDataLoader(unittest.TestCase):
    """test class of data_loader.py
    """

    def test_logScale(self):
        """test method for logScale
        """
        ndarray = np.array([1,2,3])
        min = 5
        max=6
        actual,factor,offset = scaleArray(ndarray, min, max)
        np.testing.assert_array_equal(min, np.min(actual))
        self.assertEqual(max, np.max(actual))

    def test_rescale(self):
        ndarray = np.array([1,2,3])
        min = 5
        max=6
        scaled,factor,offset = scaleArray(ndarray, min, max)
        rescaled = util.rescaleArray(scaled,factor,offset)
        np.testing.assert_array_equal(ndarray,rescaled)

    testFilePath = "dataset/train"
    def test_clip_audio_length(self):
        """
        /dataset/trainのすべての音声ファイルに対して、clipできるか確認
        :return:
        """
        print(os.getcwd())
        assert os.path.exists(self.testFilePath)
        path_list = glob.glob(os.path.join(self.testFilePath,'*.wav'))
        util.audio_dataset_second = 10
        print("test about all sound files in {}. \r\n files : {}".format(self.testFilePath,path_list.__len__()))
        for filename in path_list:
            y, sr = librosa.load(filename)
            ret = util.clip_audio_length(y,sr)
            # np.testing.assert_array_equal(ret.__len__() , util.audio_dataset_second * sr, "audioのサイズが audio_dataset_second[sec] * sr(sampling rate)[/sec]になっていない")
            self.assertEqual(ret.__len__() , util.audio_dataset_second * sr, "audio「{}」のサイズが audio_dataset_second[sec] * sr(sampling rate)[/sec]になっていない".format(filename))
            sys.stdout.write(".")

    @ unittest.skip("demonstrating skipping")
    def test_loadVoice(self):
        train_d = data_loader.Vp2pDataset("dataset/train")
#        print(list(train_d.get_example(0)))
        for i in range(3):
            label,input = train_d.get_example(i)
            plt.imshow(label[0],cmap='gray')
            plt.title(os.path.basename(train_d.get_pathName(i)))
            plt.show()
            i+=1



if __name__ == "__main__":
    unittest.main()
