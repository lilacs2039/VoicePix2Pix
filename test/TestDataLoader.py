import os
import numpy as np
import unittest
import data_loader
import matplotlib.pyplot as plt


class TestDataLoader(unittest.TestCase):
    """test class of data_loader.py
    """

    def test_logScale(self):
        """test method for logScale
        """
        ndarray = np.array([1,2,3])
        min = 5
        max=6
        actual,factor,offset = data_loader.Vp2pDataset._ScaleArray(ndarray, min, max)
        np.testing.assert_array_equal(min, np.min(actual))
        self.assertEqual(max, np.max(actual))

    def test_loadVoice(self):
        train_d = data_loader.Vp2pDataset("dataset/train")
#        print(list(train_d.get_example(0)))
        i=0
        for label,input in train_d.get_example():
            plt.imshow(label[0],cmap='gray')
            plt.title(os.path.basename(train_d.get_pathName(i)))
            plt.show()
            i+=1

if __name__ == "__main__":
    unittest.main()
