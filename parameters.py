#!/usr/bin/env python
"""
モジュール間をまたいで共有される変数を保持
"""

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


