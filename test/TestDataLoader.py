import os
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

    @unittest.skip("demonstrating skipping")
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
