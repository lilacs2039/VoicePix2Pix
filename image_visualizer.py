#!/usr/bin/env python

import os
import librosa
import numpy as np
import cupy as cp
import chainer
import chainer.cuda
from chainer import Variable
import matplotlib.pyplot as plt
import parameters


def out_image(updater, enc, dec, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        xp = enc.xp

        batch = updater.get_iterator('test').next()
        batchsize = len(batch)
        labels,inputs = xp.split(xp.asarray(batch),[1],axis=1)
        # 2次元目はinputとlabelを分けて保持していたが、分割操作により２時限目は不要になる。
        # Encoderのデータ入力形式（ndim=4）に合わせる必要あり
        # (samples,input_or_label, channel(絶対値と偏角）, 周波数, 時刻) から input_or_labelを除去する
        dataShape = list(labels.shape)
        dataShape.pop(1)
        labels = labels.reshape(dataShape)      #(1,1,2,1024,-1) -> (1,2,1024,-1)
        inputs = inputs.reshape(dataShape)
        # labels = xp.array([[[[]]]])
        # inputs = xp.array([])
        # for i in range(batchsize):
        #     #np.append(inputs,xp.asarray(batch[i][0]),axis=0)   #GPU計算の場合はarray操作はcupyで行う。cupyにはappendという関数はない
        #     xp.concatenate((inputs,xp.asarray(batch[i][0])),axis=0)
        #     xp.concatenate((labels,xp.asarray(batch[i][1])),axis=0)
        x_in = Variable(inputs)
        z = enc(x_in)
        x_out = dec(z)
        outputs = x_out.array
        if xp != np :
            #GPU計算してるなら計算後データをGPUメモリからCPU側メモリへ移動
            # cupyの対応していない関数などのせいでpyplotなどでエラーでるから
            inputs = cp.asnumpy(inputs)
            outputs = cp.asnumpy(outputs)
            labels= cp.asnumpy(labels)

        for i,input,output,label in zip(range(batchsize),inputs,outputs,labels):
            # 1figureに複数imageをプロット
            # 参考：https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly
            fig = plt.figure(figsize=(12,8))
            rows = 3
            cols=2
            def subplot(imageData,title):
                subplot.counter +=1  #1~7
                s = list(imageData.shape)
                imageData  = imageData.reshape((s[-2],s[-1]))  # (ch,h,w) -> (h,w)
                ax = fig.add_subplot(rows,cols,subplot.counter)
                plt.imshow(imageData,cmap='gray')
                ax.set_title(title)
            subplot.counter = 0
            subplot(input[0],'input abs')
            subplot(input[1],'input phase')
            subplot(output[0],'output abs')
            subplot(output[1],'output phase')
            subplot(label[0],'labels abs')
            subplot(label[1],'labels phase')

            preview_dir = '{}/preview'.format(dst)
            fileName = 'iter{:0>8}_{}'.format(trainer.updater.iteration,i)
            imagePath = os.path.join(preview_dir,fileName + '.png')
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            plt.savefig(imagePath)

            Dabs = output[0]
            Dphase = output[1]
            D = Dabs * np.exp(1j*Dphase)    #xp.exp(1j*Dphase)
            y_hat = librosa.istft(D)
            wavPath = os.path.join(preview_dir,fileName + '.wav')
            librosa.output.write_wav(wavPath,y_hat,sr=parameters.sample_ratio)

    return make_image
